"""
run: python -m src.caching.cacheEmbeddingsFromWCEB \
        --wceb src/evaluation/web-content-extraction-benchmark/datasets/combined --out cache/

Builds additional training cache from WCEB's own 8 source datasets (cetd, cleaneval,
cleanportaleval, dragnet, google-trends-2017, l3s-gn1, readability, scrapinghub), so they can
be used as extra English training data via leave-one-dataset-out (LODO) cross-validation
(see src.evaluation.aggregateLODO) rather than only in WCEB's usual benchmark-eval role.

Reuses the SAME block-parsing/labelling pipeline WCEB evaluation itself uses
(src.evaluation.blockReconstruction.parse_page / overlap_labels) instead of reimplementing
label derivation, so labels here come from WCEB's own ground-truth plaintext the identical
way evaluateBILSTM.py/evaluateXLSTM.py score a model's block-level predictions against it.

Output: cache/wceb-<dataset>-<page_id>.npz, matching the (emb, labels) .npz convention
cacheEmbeddingsForCombined.py uses for WebMainBench. The "wceb-<dataset>-" filename prefix
is what lets src.models.trainCommon.list_cache_files(..., exclude_dataset=...) filter a
fold out for LODO training -- no separate manifest file needed. Each page is parsed once
(parse_page) and the same block list feeds both overlap_labels (labels) and embed_blocks
(features), so features/labels can't drift out of alignment.

Known caveat, not fixed here: this pipeline inherits the same label/reconstruction
mismatches noted in CLAUDE.md's open investigation (truncated-label vs. untruncated
reconstruction; cell/list blocks losing their table/list wrapper) since it reuses the exact
same simplify_html/parse_page/overlap_labels machinery.
"""

import os
import argparse

import numpy as np

from src.evaluation.wcebLoader import read_wceb
from src.evaluation.blockReconstruction import (
    parse_page, block_texts, overlap_labels, overlap_labels_sequential,
)
from src.data.combinedLMEmbedder import embed_blocks

LABELERS = {"overlap": overlap_labels, "overlap_sequential": overlap_labels_sequential}


def process(html, gt_text, threshold, min_words, labeler_name="overlap"):
    simpl_blocks, _mapping_blocks = parse_page(html)
    if not simpl_blocks:
        return None, "no-blocks"
    texts = block_texts(simpl_blocks)
    labels = LABELERS[labeler_name](texts, gt_text, threshold, min_words)
    emb = np.asarray(embed_blocks(simpl_blocks), dtype=np.float32)
    return (emb, np.asarray(labels, dtype=np.float32)), f"wceb-{labeler_name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wceb", required=True, help="path to .../datasets/combined")
    ap.add_argument("--datasets", nargs="*", default=None, help="subset names (default: all 8)")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--labeler", choices=list(LABELERS), default="overlap",
                    help="overlap: word-overlap heuristic (default, unchanged legacy behavior). "
                         "overlap_sequential: position-aware labeler, see "
                         "analysis/oracle_investigation.md Part 5 -- significantly better on "
                         "5/8 WCEB datasets but significantly WORSE on cleaneval, so a full "
                         "rebuild should run cleaneval separately with --labeler overlap")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="min fraction of a block's tokens that must be in the GT vocab")
    ap.add_argument("--min-words", type=int, default=3,
                    help="blocks shorter than this need a full token match, not just threshold")
    ap.add_argument("--limit", type=int, default=0, help="process only first N pages (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"wceb={args.wceb}  out={args.out}  labeler={args.labeler}  "
          f"threshold={args.threshold}  min_words={args.min_words}")

    kept = skipped = 0
    total_blocks = total_pos = 0
    per_dataset = {}

    for i, (ds, page_id, html, gt) in enumerate(read_wceb(args.wceb, args.datasets)):
        if args.limit and kept >= args.limit:
            break

        path = os.path.join(args.out, f"wceb-{ds}-{page_id}.npz")
        if os.path.exists(path):
            continue

        try:
            out, info = process(html, gt, args.threshold, args.min_words, args.labeler)
        except Exception as e:
            print(f"[{ds}/{page_id[:8]}] skip ({type(e).__name__}): {e}")
            skipped += 1
            continue

        if out is None:
            skipped += 1
            continue

        emb, labels = out
        np.savez(path, emb=emb, labels=labels, label_method=info)
        kept += 1
        total_blocks += len(labels)
        total_pos += int(labels.sum())
        per_dataset[ds] = per_dataset.get(ds, 0) + 1

        if kept % 200 == 0:
            pos_rate = total_pos / max(total_blocks, 1)
            print(f"[{i}] cached={kept} skipped={skipped}  overall pos-rate={pos_rate:.3f}")

    print("-" * 60)
    print(f"done. cached={kept} skipped={skipped}")
    for ds, n in sorted(per_dataset.items()):
        print(f"  {ds:20s} pages={n}")
    if total_blocks:
        print(f"overall positive block rate: {total_pos / total_blocks:.3f}")
        if total_pos == 0:
            print("  WARNING: zero positive labels -> labeling is broken, do NOT train on this.")


if __name__ == "__main__":
    main()
