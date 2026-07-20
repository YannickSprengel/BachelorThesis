"""
Benchmark the trained model on WCEB (Bevendorff et al., 2023), reporting three things:

  1. ROUGE-5 F1 and ROUGE-L F1 (jieba)
  2. Block-level Precision / Recall / F1 of the model's keep/drop decisions   [model only]
  3. Extraction throughput: pages/sec + mean/median sec per page, with hardware/device.
        Only the extraction (simplify + embed + predict + reconstruct) is timed
        file I/O, text conversion and metric computation are excluded.

    How to run: python -m src.evaluation.evaluateBILSTM --model model.pt \
        --wceb src/evaluation/wceb_data/combined --out results/wceb

See src.evaluation.evaluateOracleCeiling for a model-free version of this that
reconstructs from gold labels instead of model predictions, to check how much of the
score ceiling comes from preprocessing/reconstruction rather than the tagger.
"""

import argparse
import csv
import json
import os
import time
import platform
import statistics
from collections import defaultdict
import torch

from src.evaluation.wcebLoader import read_wceb
from src.evaluation.textMetrics import rouge_n_f1, rouge_l_f1, to_text
from src.evaluation.blockReconstruction import parse_page, block_texts, reconstruct, overlap_labels, prf
from src.data.combinedLMEmbedder import embed_blocks
from src.models.lstm.biLSTMWithMiniLM import BiLSTMTagger


@torch.no_grad()
def predict_page(model, html, device, threshold=0.5):
    """One simplify pass -> (body_html, keep mask, per-block text). This call is the 'extraction'."""
    simpl, mapping = parse_page(html)
    emb = torch.as_tensor(embed_blocks(simpl), dtype=torch.float32, device=device).unsqueeze(0)
    keep = (torch.sigmoid(model(emb)).squeeze(0).cpu() > threshold)   # .cpu() forces materialisation
    body = reconstruct(simpl, mapping, keep)
    return body, keep, block_texts(simpl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--wceb", required=True, help="path to .../datasets/combined")
    ap.add_argument("--datasets", nargs="*", default=None, help="subset names (default: all)")
    ap.add_argument("--n", type=int, default=5)              # ROUGE-N, paper uses N=5
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="model keep-decision threshold (sigmoid cutoff); this is the one to tune")
    ap.add_argument("--label-threshold", type=float, default=0.5,
                    help="FIXED overlap cutoff for the silver labels; keep constant when tuning --threshold")
    ap.add_argument("--out", default="wceb_results", help="output basename for .csv and .json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hw = torch.cuda.get_device_name(0) if device == "cuda" \
        else f"{platform.processor() or platform.machine()} x{os.cpu_count()} (CPU)"

    model = BiLSTMTagger().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  hw={hw}  BiLSTM params={n_params:,}")
    print(f"keep-threshold={args.threshold}  label-threshold(silver)={args.label_threshold}")
    print(f"(ROUGE-{args.n} F1 + ROUGE-L F1, jieba  +  block-level P/R/F1  +  throughput)")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    csv_file = open(args.out + ".csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["dataset", "page_id", "rouge5", "rouge_l", "n_blocks", "n_kept",
                     "pred_chars", "gt_chars", "tp", "fp", "fn", "sec"])

    scores5, scoresL = [], []
    by_ds5, by_dsL = defaultdict(list), defaultdict(list)
    times = []
    TP = FP = FN = 0
    ds_blocks = defaultdict(lambda: [0, 0, 0])     # dataset -> [tp, fp, fn]
    skipped = 0

    for ds, page_id, html, gt in read_wceb(args.wceb, args.datasets):
        try:
            t0 = time.perf_counter()
            body, keep, texts = predict_page(model, html, device, args.threshold)
            dt = time.perf_counter() - t0          # extraction time only
            pred_text = to_text(body)              # not timed (eval/metric step)
        except Exception as e:
            print(f"[{ds}/{page_id[:8]}] skip: {e}")
            skipped += 1
            continue

        s5 = rouge_n_f1(pred_text, gt, n=args.n)
        sL = rouge_l_f1(pred_text, gt)

        # block-level: model decision vs silver overlap label
        gt_lab = overlap_labels(texts, gt, args.label_threshold)
        pred_lab = [int(k) for k in keep]
        tp = sum(p == 1 and g == 1 for p, g in zip(pred_lab, gt_lab))
        fp = sum(p == 1 and g == 0 for p, g in zip(pred_lab, gt_lab))
        fn = sum(p == 0 and g == 1 for p, g in zip(pred_lab, gt_lab))
        TP += tp; FP += fp; FN += fn
        ds_blocks[ds][0] += tp; ds_blocks[ds][1] += fp; ds_blocks[ds][2] += fn

        scores5.append(s5); scoresL.append(sL)
        by_ds5[ds].append(s5); by_dsL[ds].append(sL)
        times.append(dt)
        writer.writerow([ds, page_id, f"{s5:.6f}", f"{sL:.6f}", len(keep), int(keep.sum()),
                         len(pred_text), len(gt), tp, fp, fn, f"{dt:.4f}"])
        csv_file.flush()
        if len(scores5) % 100 == 0:
            print(f"  {len(scores5)} docs  rouge5_mean={sum(scores5)/len(scores5):.4f}  "
                  f"rougeL_mean={sum(scoresL)/len(scoresL):.4f}  "
                  f"sec/page_median={statistics.median(times):.3f}")

    csv_file.close()

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    bP, bR, bF = prf(TP, FP, FN)
    summary = {
        "model": args.model,
        "device": device, "hardware": hw, "bilstm_params": n_params,
        "keep_threshold": args.threshold,
        "label_threshold": args.label_threshold,
        "rouge5": {
            "n": args.n,
            "n_docs": len(scores5), "n_skipped": skipped,
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
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== WCEB ROUGE-{args.n} F1 / ROUGE-L F1 (jieba) ===")
    print(f"Overall ({len(scores5):4d}, skipped {skipped}): "
          f"rouge{args.n}={mean(scores5):.4f}  rougeL={mean(scoresL):.4f}")
    for ds in sorted(by_ds5):
        print(f"  {ds:20s} ({len(by_ds5[ds]):4d}): "
              f"rouge{args.n}={mean(by_ds5[ds]):.4f}  rougeL={mean(by_dsL[ds]):.4f}")

    print(f"\n=== Block-level (vs silver overlap labels) ===")
    print(f"Precision {bP:.4f}  Recall {bR:.4f}  F1 {bF:.4f}   (tp={TP} fp={FP} fn={FN})")

    print(f"\n=== Throughput on {device} ({hw}) ===")
    if times:
        print(f"pages/sec {len(times)/sum(times):.3f}   "
              f"sec/page mean {statistics.mean(times):.4f}  median {statistics.median(times):.4f}")
    print(f"\nsaved: {args.out}.csv  +  {args.out}.json")


if __name__ == "__main__":
    main()
