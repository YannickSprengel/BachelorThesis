"""
Shared, architecture-agnostic WCEB evaluation loop used by evaluateBILSTM.py, evaluateXLSTM.py,
and (in-process, no subprocess) by sweep.py's top-K stage and aggregateLODO.py's per-fold eval.

predict_page only depends on model(emb) via a forward pass shared by BiLSTMTagger/XLSTMTagger,
so one implementation covers both taggers.
"""

import csv
import json
import os
import platform
import statistics
from collections import defaultdict

import torch

from src.evaluation.wcebLoader import read_wceb
from src.evaluation.textMetrics import rouge_n_f1, rouge_l_f1, to_text
from src.evaluation.blockReconstruction import parse_page, block_texts, reconstruct, overlap_labels, prf
from src.data.combinedLMEmbedder import embed_blocks


@torch.no_grad()
def predict_page(model, html, device, threshold=0.5):
    """One simplify pass -> (body_html, keep mask, per-block text). This call is the 'extraction'."""
    simpl, mapping = parse_page(html)
    emb = torch.as_tensor(embed_blocks(simpl), dtype=torch.float32, device=device).unsqueeze(0)
    keep = (torch.sigmoid(model(emb)).squeeze(0).cpu() > threshold)   # .cpu() forces materialisation
    body = reconstruct(simpl, mapping, keep)
    return body, keep, block_texts(simpl)


def _hardware_label(device):
    return torch.cuda.get_device_name(0) if device == "cuda" \
        else f"{platform.processor() or platform.machine()} x{os.cpu_count()} (CPU)"


def run_eval(model, wceb_dir, datasets=None, n=5, threshold=0.5, label_threshold=0.5,
             device="cpu", model_path=None, arch=None, out_basename=None):
    """Runs the simplify->embed->predict->reconstruct pipeline over WCEB, scoring ROUGE-5/L
    against ground truth and block-level P/R/F1 against silver overlap labels, plus throughput.
    If out_basename is given, also writes <out_basename>.csv / .json exactly as evaluate*.py do.
    Returns the summary dict either way."""
    n_params = sum(p.numel() for p in model.parameters())
    hw = _hardware_label(device)

    writer = csv_file = None
    if out_basename:
        out_dir = os.path.dirname(out_basename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        csv_file = open(out_basename + ".csv", "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["dataset", "page_id", "rouge5", "rouge_l", "n_blocks", "n_kept",
                         "pred_chars", "gt_chars", "tp", "fp", "fn", "sec"])

    scores5, scoresL = [], []
    by_ds5, by_dsL = defaultdict(list), defaultdict(list)
    times = []
    TP = FP = FN = 0
    ds_blocks = defaultdict(lambda: [0, 0, 0])
    skipped = 0

    import time as _time
    for ds, page_id, html, gt in read_wceb(wceb_dir, datasets):
        try:
            t0 = _time.perf_counter()
            body, keep, texts = predict_page(model, html, device, threshold)
            dt = _time.perf_counter() - t0
            pred_text = to_text(body)
        except Exception as e:
            print(f"[{ds}/{page_id[:8]}] skip: {e}")
            skipped += 1
            continue

        s5 = rouge_n_f1(pred_text, gt, n=n)
        sL = rouge_l_f1(pred_text, gt)

        gt_lab = overlap_labels(texts, gt, label_threshold)
        pred_lab = [int(k) for k in keep]
        tp = sum(p == 1 and g == 1 for p, g in zip(pred_lab, gt_lab))
        fp = sum(p == 1 and g == 0 for p, g in zip(pred_lab, gt_lab))
        fn = sum(p == 0 and g == 1 for p, g in zip(pred_lab, gt_lab))
        TP += tp; FP += fp; FN += fn
        ds_blocks[ds][0] += tp; ds_blocks[ds][1] += fp; ds_blocks[ds][2] += fn

        scores5.append(s5); scoresL.append(sL)
        by_ds5[ds].append(s5); by_dsL[ds].append(sL)
        times.append(dt)
        if writer:
            writer.writerow([ds, page_id, f"{s5:.6f}", f"{sL:.6f}", len(keep), int(keep.sum()),
                             len(pred_text), len(gt), tp, fp, fn, f"{dt:.4f}"])
            csv_file.flush()
        if len(scores5) % 100 == 0:
            print(f"  {len(scores5)} docs  rouge5_mean={sum(scores5)/len(scores5):.4f}  "
                  f"rougeL_mean={sum(scoresL)/len(scoresL):.4f}  "
                  f"sec/page_median={statistics.median(times):.3f}", flush=True)

    if csv_file:
        csv_file.close()

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    bP, bR, bF = prf(TP, FP, FN)
    summary = {
        "model": model_path, "arch": arch,
        "device": device, "hardware": hw, "n_params": n_params,
        "keep_threshold": threshold, "label_threshold": label_threshold,
        "rouge5": {
            "n": n, "n_docs": len(scores5), "n_skipped": skipped,
            "overall_f1": round(mean(scores5), 6),
            "by_dataset": {ds: {"n": len(by_ds5[ds]), "f1": round(mean(by_ds5[ds]), 6)}
                           for ds in sorted(by_ds5)},
        },
        "rouge_l": {
            "n_docs": len(scoresL), "n_skipped": skipped,
            "overall_f1": round(mean(scoresL), 6),
            "by_dataset": {ds: {"n": len(by_dsL[ds]), "f1": round(mean(by_dsL[ds]), 6)}
                           for ds in sorted(by_dsL)},
        },
        "block_level": {
            "note": "silver labels (token overlap with GT text)",
            "precision": round(bP, 4), "recall": round(bR, 4), "f1": round(bF, 4),
            "tp": TP, "fp": FP, "fn": FN,
            "by_dataset": {ds: {"f1": round(prf(*c)[2], 4)} for ds, c in sorted(ds_blocks.items())},
        },
        "throughput": {
            "pages": len(times),
            "total_extract_sec": round(sum(times), 2),
            "pages_per_sec": round(len(times) / sum(times), 3) if times else 0.0,
            "sec_per_page_mean": round(statistics.mean(times), 4) if times else 0.0,
            "sec_per_page_median": round(statistics.median(times), 4) if times else 0.0,
        },
    }

    if out_basename:
        with open(out_basename + ".json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def print_report(summary):
    n, skipped = summary["rouge5"]["n"], summary["rouge5"]["n_skipped"]
    scores5, scoresL = summary["rouge5"], summary["rouge_l"]
    print(f"\n=== WCEB ROUGE-{n} F1 / ROUGE-L F1 (jieba) ===")
    print(f"Overall ({scores5['n_docs']:4d}, skipped {skipped}): "
          f"rouge{n}={scores5['overall_f1']:.4f}  rougeL={scoresL['overall_f1']:.4f}")
    for ds in sorted(scores5["by_dataset"]):
        d5, dL = scores5["by_dataset"][ds], scoresL["by_dataset"][ds]
        print(f"  {ds:20s} ({d5['n']:4d}): rouge{n}={d5['f1']:.4f}  rougeL={dL['f1']:.4f}")

    bl = summary["block_level"]
    print(f"\n=== Block-level (vs silver overlap labels) ===")
    print(f"Precision {bl['precision']:.4f}  Recall {bl['recall']:.4f}  F1 {bl['f1']:.4f}   "
          f"(tp={bl['tp']} fp={bl['fp']} fn={bl['fn']})")

    tp_ = summary["throughput"]
    print(f"\n=== Throughput on {summary['device']} ({summary['hardware']}) ===")
    if tp_["pages"]:
        print(f"pages/sec {tp_['pages_per_sec']:.3f}   "
              f"sec/page mean {tp_['sec_per_page_mean']:.4f}  median {tp_['sec_per_page_median']:.4f}")
