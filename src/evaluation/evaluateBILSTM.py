"""
evaluate_wceb.py
================
Benchmark the trained model on WCEB (Bevendorff et al., 2023), reporting three things:

  1. ROUGE-N F1 (N=5, jieba)
  2. Block-level Precision / Recall / F1 of the model's keep/drop decisions   [model only]
  3. Extraction throughput: pages/sec + mean/median sec per page, with hardware/device.
        Only the extraction (simplify + embed + predict + reconstruct) is timed
        file I/O, text conversion and metric computation are excluded.

    How to run: python -m src.evaluation.evaluateBILSTM --model model.pt \
        --wceb src/evaluation/wceb_data/combined --out results/wceb

"""

import argparse
import csv
import json
import os
import re
import time
import platform
import statistics
from collections import Counter, defaultdict
import torch
import jieba

from src.evaluation.wcebLoader import read_wceb
from src.data.combinedLMEmbedder import simplify_html, _parse_blocks, embed_blocks
from src.models.lstm.biLSTMWithMiniLM import BiLSTMTagger

_TOK = re.compile(r"\w+", re.UNICODE)


def to_text(html):
    """HTML -> plain text. Uses html-text (what Dripper used for WCEB); falls back to bs4."""
    try:
        import html_text
        return html_text.extract_text(html or "")
    except ImportError:
        from bs4 import BeautifulSoup
        return BeautifulSoup(html or "", "html.parser").get_text(" ", strip=True)


def rouge_n_f1(pred, ref, n=5):
    pt, rt = jieba.lcut(pred or ""), jieba.lcut(ref or "")
    ng = lambda t: Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
    pg, rg = ng(pt), ng(rt)
    if not pg or not rg:
        return 0.0
    overlap = sum((pg & rg).values())          # clipped n-gram overlap
    if overlap == 0:
        return 0.0
    prec, rec = overlap / sum(pg.values()), overlap / sum(rg.values())
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def predict_page(model, html, device, threshold=0.5):
    """One simplify pass -> (body_html, keep mask, per-block text). This call is the 'extraction'."""
    simp_str, map_str = simplify_html(html)
    simpl = _parse_blocks(simp_str)              # what the model sees / is labelled on
    mapping = _parse_blocks(map_str)             # original DOM, used to rebuild content
    emb = torch.as_tensor(embed_blocks(simpl), dtype=torch.float32, device=device).unsqueeze(0)
    keep = (torch.sigmoid(model(emb)).squeeze(0).cpu() > threshold)   # .cpu() forces materialisation

    # Reconstruct from the mapping branch, matched by _item_id
    map_by_id = {b.get("_item_id"): b for b in mapping}
    kept = [map_by_id[b.get("_item_id")]
            for b, k in zip(simpl, keep)
            if k and b.get("_item_id") in map_by_id]

    # Drop nested duplicates: if an ancestor of a kept block is itself kept
    kept_ids = {id(b) for b in kept}
    top = [b for b in kept if not any(id(a) in kept_ids for a in b.parents)]

    body = "\n".join(str(b) for b in top)
    block_texts = [b.get_text(" ", strip=True) for b in simpl]        # text basis for silver labels
    return body, keep, block_texts


def overlap_labels(block_texts, gt, threshold=0.5):
    """Silver block label: 1 if >= threshold of a block's tokens appear in the GT text (as in training)."""
    gt_tokens = {t.lower() for t in _TOK.findall(gt or "")}
    out = []
    for bt in block_texts:
        toks = [t.lower() for t in _TOK.findall(bt or "")]
        if not toks:
            out.append(0)
            continue
        frac = sum(1 for t in toks if t in gt_tokens) / len(toks)
        out.append(1 if frac >= threshold else 0)
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


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
    print(f"(ROUGE-{args.n} F1 jieba Html+TEXT  +  block-level P/R/F1  +  throughput)")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    csv_file = open(args.out + ".csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["dataset", "page_id", "rouge", "n_blocks", "n_kept",
                     "pred_chars", "gt_chars", "tp", "fp", "fn", "sec"])

    scores, by_ds, times = [], defaultdict(list), []
    TP = FP = FN = 0
    ds_blocks = defaultdict(lambda: [0, 0, 0])     # dataset -> [tp, fp, fn]
    skipped = 0

    for ds, page_id, html, gt in read_wceb(args.wceb, args.datasets):
        try:
            t0 = time.perf_counter()
            body, keep, block_texts = predict_page(model, html, device, args.threshold)
            dt = time.perf_counter() - t0          # extraction time only
            pred_text = to_text(body)              # not timed (eval/metric step)
        except Exception as e:
            print(f"[{ds}/{page_id[:8]}] skip: {e}")
            skipped += 1
            continue

        s = rouge_n_f1(pred_text, gt, n=args.n)

        # block-level: model decision vs silver overlap label
        # IMPORTANT: the silver labels use a FIXED overlap cutoff (args.label_threshold),
        # NOT the model keep-threshold. Coupling them would move the ground-truth definition
        # whenever you tune --threshold, making block-level P/R/F1 uninterpretable.
        gt_lab = overlap_labels(block_texts, gt, args.label_threshold)
        pred_lab = [int(k) for k in keep]
        tp = sum(p == 1 and g == 1 for p, g in zip(pred_lab, gt_lab))
        fp = sum(p == 1 and g == 0 for p, g in zip(pred_lab, gt_lab))
        fn = sum(p == 0 and g == 1 for p, g in zip(pred_lab, gt_lab))
        TP += tp; FP += fp; FN += fn
        ds_blocks[ds][0] += tp; ds_blocks[ds][1] += fp; ds_blocks[ds][2] += fn

        scores.append(s); by_ds[ds].append(s); times.append(dt)
        writer.writerow([ds, page_id, f"{s:.6f}", len(keep), int(keep.sum()),
                         len(pred_text), len(gt), tp, fp, fn, f"{dt:.4f}"])
        csv_file.flush()
        if len(scores) % 100 == 0:
            print(f"  {len(scores)} docs  rouge_mean={sum(scores)/len(scores):.4f}  "
                  f"sec/page_median={statistics.median(times):.3f}")

    csv_file.close()

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    bP, bR, bF = prf(TP, FP, FN)
    summary = {
        "model": args.model,
        "device": device, "hardware": hw, "bilstm_params": n_params,
        "keep_threshold": args.threshold,
        "label_threshold": args.label_threshold,
        "rouge": {
            "n": args.n,
            "n_docs": len(scores), "n_skipped": skipped,
            "overall_f1": round(mean(scores), 6),
            "by_dataset": {ds: {"n": len(by_ds[ds]), "f1": round(mean(by_ds[ds]), 6)}
                           for ds in sorted(by_ds)},
        },
        "block_level": {
            "note": "silver labels (token overlap with GT text); proxy, not human gold",
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

    print(f"\n=== WCEB ROUGE-{args.n} F1 (Html+TEXT, jieba) ===")
    print(f"Overall ({len(scores):4d}, skipped {skipped}): {mean(scores):.4f}")
    for ds in sorted(by_ds):
        print(f"  {ds:20s} ({len(by_ds[ds]):4d}): {mean(by_ds[ds]):.4f}")

    print(f"\n=== Block-level (vs silver overlap labels) ===")
    print(f"Precision {bP:.4f}  Recall {bR:.4f}  F1 {bF:.4f}   (tp={TP} fp={FP} fn={FN})")

    print(f"\n=== Throughput on {device} ({hw}) ===")
    if times:
        print(f"pages/sec {len(times)/sum(times):.3f}   "
              f"sec/page mean {statistics.mean(times):.4f}  median {statistics.median(times):.4f}")
    print(f"\nsaved: {args.out}.csv  +  {args.out}.json")


if __name__ == "__main__":
    main()