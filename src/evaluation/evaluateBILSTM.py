"""
evaluate_wceb.py
================
Benchmark the trained model on WCEB (Bevendorff et al., 2023) in "Html+TEXT" mode
(Dripper §5.4):  extracted HTML -> plain text -> ROUGE-N F1 (N=5, jieba) vs plaintext.

Saves results:
  <out>.csv   one row per page (written incrementally, survives a crash)
  <out>.json  summary: overall + per sub-dataset means
and still prints the summary to the console.

    python -m src.evaluation.evaluate_wceb --model model.pt \
        --wceb src/evaluation/wceb_data/combined --out results/wceb

Requires: jieba, html-text   (plus the training/inference deps)
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
import torch
import jieba

from src.evaluation.wcebLoader import read_wceb
from src.models.lstm.biLSTMWithMiniLM import BiLSTMTagger
from src.models.lstm.tryout import extract_main_content


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--wceb", required=True, help="path to .../datasets/combined")
    ap.add_argument("--datasets", nargs="*", default=None, help="subset names (default: all)")
    ap.add_argument("--n", type=int, default=5)            # ROUGE-N, paper uses N=5
    ap.add_argument("--out", default="wceb_results", help="output basename for .csv and .json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BiLSTMTagger().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"device={device}  (ROUGE-{args.n} F1, jieba, Html+TEXT)")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # per-page CSV, written and flushed row by row so a crash keeps partial results
    csv_file = open(args.out + ".csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["dataset", "page_id", "rouge", "n_blocks", "n_kept", "pred_chars", "gt_chars"])

    scores, by_ds = [], defaultdict(list)
    skipped = 0
    for ds, page_id, html, gt in read_wceb(args.wceb, args.datasets):
        try:
            body, keep = extract_main_content(model, html, device=device)
            pred = to_text(body)
        except Exception as e:
            print(f"[{ds}/{page_id[:8]}] skip: {e}")
            skipped += 1
            continue
        s = rouge_n_f1(pred, gt, n=args.n)
        scores.append(s)
        by_ds[ds].append(s)
        writer.writerow([ds, page_id, f"{s:.6f}", len(keep), int(keep.sum()), len(pred), len(gt)])
        csv_file.flush()
        if len(scores) % 100 == 0:
            print(f"  {len(scores)} docs  running mean={sum(scores)/len(scores):.4f}")

    csv_file.close()

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    summary = {
        "model": args.model,
        "rouge_n": args.n,
        "n_docs": len(scores),
        "n_skipped": skipped,
        "overall_f1": round(mean(scores), 6),
        "by_dataset": {ds: {"n": len(by_ds[ds]), "f1": round(mean(by_ds[ds]), 6)}
                       for ds in sorted(by_ds)},
    }
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== WCEB ROUGE-{args.n} F1 (Html+TEXT, jieba) ===")
    print(f"Overall ({len(scores):4d}, skipped {skipped}): {mean(scores):.4f}")
    for ds in sorted(by_ds):
        print(f"  {ds:20s} ({len(by_ds[ds]):4d}): {mean(by_ds[ds]):.4f}")
    print(f"\nsaved: {args.out}.csv  +  {args.out}.json")


if __name__ == "__main__":
    main()