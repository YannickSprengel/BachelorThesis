"""
In-memory training data for the trainable-MiniLM path: parsed blocks + labels, kept as plain
Python objects instead of precomputed embedding vectors, since the embedding step now happens
inside the model's forward pass (see src.data.trainableMiniLM.TrainableMiniLMEmbedder) and can't
be cached ahead of time.

DOM segmentation and label derivation are still deterministic and cheap relative to a MiniLM
forward/backward pass, so they run once here (not once per epoch) and stay in RAM for the whole
run, mirroring what src.models.trainCommon.load_cache_to_memory does for the .npz caches.

Reuses the exact label-derivation code cacheEmbeddingsForCombined.py uses for WebMainBench, so
labels here match what a frozen-embedding run would have cached for the same page.
"""

import numpy as np
import torch

from src.caching.cacheEmbeddingsForCombined import (
    groundtruth_words, overlap_labels, iter_jsonl,
)
from src.data.combinedLMEmbedder import simplify_html, _parse_blocks


def load_webmainbench_blocks(jsonl_path, threshold=0.5, min_words=3, limit=0):
    """Returns a list of (blocks, labels) pairs: blocks is a list[bs4.Tag] for one page
    (TrainableMiniLMEmbedder's expected input), labels is a (len(blocks),) float32 tensor."""
    data = []
    skipped = 0
    for row in iter_jsonl(jsonl_path):
        if limit and len(data) >= limit:
            break

        html = row.get("html")
        if not html:
            skipped += 1
            continue

        gt_words, source = groundtruth_words(row)
        if not gt_words:
            skipped += 1
            continue

        try:
            simplified_html_str, _mapping = simplify_html(html)
            blocks = _parse_blocks(simplified_html_str)
        except Exception as e:
            print(f"skip ({type(e).__name__}): {e}")
            skipped += 1
            continue

        if not blocks:
            skipped += 1
            continue

        labels = overlap_labels(blocks, gt_words, threshold, min_words)
        data.append((blocks, torch.from_numpy(np.asarray(labels, dtype=np.float32))))

    print(f"loaded {len(data)} pages, skipped {skipped} (no html/no label source/no blocks)")
    return data


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    data = load_webmainbench_blocks(args.jsonl, limit=args.limit)
    if data:
        blocks, labels = data[0]
        print(f"first page: {len(blocks)} blocks, labels shape={tuple(labels.shape)}")
