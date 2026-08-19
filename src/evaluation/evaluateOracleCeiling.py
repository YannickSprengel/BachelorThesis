"""
Oracle ceiling check: reconstruct WCEB pages using the GOLD keep/drop labels (the same
token-overlap heuristic used to build training labels and to score block-level P/R/F1
in evaluateBILSTM.py / evaluateXLSTM.py) instead of a trained model's predictions, then
score the reconstructed text against WCEB ground truth. No model, no embeddings.

This isolates the preprocessing/reconstruction pipeline from the tagger: if this
oracle's ROUGE stays low, no amount of model training will fix the plateau, the
pipeline itself is the ceiling. If it's close to 1.0, the pipeline is fine and a low
model score means the tagger needs work, not the reconstruction.

run: python -m src.evaluation.evaluateOracleCeiling --wceb src/evaluation/wceb_data/combined --out results/oracle
"""

import argparse
import csv
import json
import os
from collections import defaultdict

from src.evaluation.wcebLoader import read_wceb
from src.evaluation.textMetrics import rouge_n_f1, rouge_l_f1, to_text
from src.evaluation.blockReconstruction import (
    parse_page, block_texts, reconstruct, overlap_labels, overlap_labels_weighted,
    overlap_labels_sequential,
)

LABELERS = {
    "overlap": overlap_labels,
    "overlap_weighted": overlap_labels_weighted,
    "overlap_sequential": overlap_labels_sequential,
}


def oracle_page(html, gt, threshold, min_words, labeler=overlap_labels):
    """Same shape as evaluate*.py's predict_page, but the keep mask comes directly
    from the gold overlap labels instead of a model."""
    simpl, mapping = parse_page(html)
    texts = block_texts(simpl)
    keep = labeler(texts, gt, threshold, min_words)
    body = reconstruct(simpl, mapping, keep)
    return body, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wceb", required=True, help="path to .../datasets/combined")
    ap.add_argument("--datasets", nargs="*", default=None, help="subset names (default: all)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="token-overlap cutoff used to build the oracle's keep/drop labels")
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--labeler", choices=list(LABELERS), default="overlap",
                    help="overlap: set-membership token overlap (default). overlap_weighted: "
                         "frequency-aware clipped-multiset overlap, see blockReconstruction.py")
    ap.add_argument("--out", default="oracle_results", help="output basename for .csv and .json")
    args = ap.parse_args()
    labeler = LABELERS[args.labeler]

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    csv_file = open(args.out + ".csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["dataset", "page_id", "rouge5", "rouge_l", "n_blocks", "n_kept",
                     "pred_chars", "gt_chars"])

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    scores5, scoresL = [], []
    by_ds5, by_dsL = defaultdict(list), defaultdict(list)
    skipped = 0

    for ds, page_id, html, gt in read_wceb(args.wceb, args.datasets):
        try:
            body, keep = oracle_page(html, gt, args.threshold, args.min_words, labeler)
            pred_text = to_text(body)
        except Exception as e:
            print(f"[{ds}/{page_id[:8]}] skip: {e}")
            skipped += 1
            continue

        s5 = rouge_n_f1(pred_text, gt, n=args.n)
        sL = rouge_l_f1(pred_text, gt)

        scores5.append(s5); scoresL.append(sL)
        by_ds5[ds].append(s5); by_dsL[ds].append(sL)
        writer.writerow([ds, page_id, f"{s5:.6f}", f"{sL:.6f}", len(keep), sum(keep),
                         len(pred_text), len(gt)])
        csv_file.flush()
        if len(scores5) % 100 == 0:
            print(f"  {len(scores5)} docs  rouge5_mean={mean(scores5):.4f}  rougeL_mean={mean(scoresL):.4f}")

    csv_file.close()

    summary = {
        "keep_source": "oracle (gold token-overlap labels, no model)",
        "labeler": args.labeler,
        "label_threshold": args.threshold,
        "rouge5": {
            "n": args.n, "n_docs": len(scores5), "n_skipped": skipped,
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
    }
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Oracle ceiling (gold keep/drop labels, overlap threshold={args.threshold}) ===")
    print(f"ROUGE-{args.n}  ({len(scores5):4d}, skipped {skipped}): {mean(scores5):.4f}")
    print(f"ROUGE-L  ({len(scoresL):4d}, skipped {skipped}): {mean(scoresL):.4f}")
    for ds in sorted(by_ds5):
        print(f"  {ds:20s}  rouge{args.n}={mean(by_ds5[ds]):.4f}  rougeL={mean(by_dsL[ds]):.4f}")
    print(f"\nsaved: {args.out}.csv  +  {args.out}.json")
    print("\nThis is the score ceiling if the tagger made every keep/drop decision perfectly")
    print("(by the same overlap heuristic the model is trained on). A trained model can only")
    print("approach this number, never beat it, so compare it directly to your model's ROUGE runs.")


if __name__ == "__main__":
    main()
