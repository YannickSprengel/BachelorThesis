"""
run:    python cache_embeddings.py --jsonl data/webmainbench.jsonl --out cache/

The ground-truth main content in WebMainBench is annotated *inside* the `html`
field: the human-selected regions carry the attribute  cc-select="true".  That
attribute is the most reliable label source, because `html` is always present
(unlike groundtruth_content, which only exists for the 545-sample subset).

So for each page we:
    1. collect the words inside every cc-select="true" region of the ORIGINAL html
       -> this is the ground-truth content vocabulary
    2. parse the blocks the model will see       (simplify_html -> _parse_blocks)
    3. label a block 1 if >= `threshold` of its words appear in that vocabulary

If a page has no cc-select regions, we fall back to convert_main_content /
groundtruth_content / main_html, in that order.
"""

import os
import re
import json
import argparse

import numpy as np
from bs4 import BeautifulSoup

from src.data.combinedLMEmbedder import simplify_html, _parse_blocks, embed_document

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    return _WORD.findall((text or "").lower())


def block_words(block):
    """Words of a parsed block, whether it's a BeautifulSoup Tag or a plain string."""
    if hasattr(block, "get_text"):
        return tokenize(block.get_text(" ", strip=True))
    return tokenize(str(block))


# ---------------------------------------------------------------------------
# Ground-truth vocabulary
# ---------------------------------------------------------------------------

def cc_select_words(html):
    """Set of words inside all cc-select="true" regions of the original html.

    Returns (word_set, n_nodes). get_text() on a marked node already includes its
    descendants, so nested marks are covered by the union.
    """
    soup = BeautifulSoup(html, "html.parser")
    nodes = [n for n in soup.find_all(attrs={"cc-select": True})
             if str(n.get("cc-select")).lower() == "true"]
    words = set()
    for n in nodes:
        words.update(tokenize(n.get_text(" ", strip=True)))
    return words, len(nodes)


def groundtruth_words(row):
    """(word_set, source_name). Prefer the cc-select annotation; fall back to text fields."""
    words, n = cc_select_words(row.get("html") or "")
    if n > 0 and words:
        return words, "cc-select"

    for field in ("convert_main_content", "groundtruth_content"):
        txt = row.get(field)
        if txt:
            return set(tokenize(txt)), field

    main_html = row.get("main_html")
    if main_html:
        text = BeautifulSoup(main_html, "html.parser").get_text(" ", strip=True)
        return set(tokenize(text)), "main_html"

    return set(), None


def overlap_labels(blocks, gt_words, threshold, min_words):
    """1 if >= threshold of a block's words appear in the ground-truth vocabulary."""
    labels = np.zeros(len(blocks), dtype=np.float32)
    for i, b in enumerate(blocks):
        w = block_words(b)
        if len(w) < min_words:
            continue
        frac = sum(t in gt_words for t in w) / len(w)
        if frac >= threshold:
            labels[i] = 1.0
    return labels


# ---------------------------------------------------------------------------
# Per-document processing
# ---------------------------------------------------------------------------

def process(row, threshold, min_words):
    """Returns ((emb, labels), source) or (None, reason)."""
    html = row.get("html")
    if not html:
        return None, "no-html"

    gt_words, source = groundtruth_words(row)
    if not gt_words:
        return None, "no-label-source"

    # embed_document takes the RAW html: it runs simplify_html + _parse_blocks
    # internally and encodes str(block), so tags/structure are kept in the vector.
    emb = np.asarray(embed_document(html), dtype=np.float32)   # (n, 384)

    # Re-derive the SAME blocks (identical _parse_blocks / _item_id order) so we can
    # attach exactly one label per embedded block.
    simplified_html_str, _mapping = simplify_html(html)
    blocks = _parse_blocks(simplified_html_str)
    if not blocks:
        return None, "no-blocks"

    if emb.shape[0] != len(blocks):
        return None, f"align-mismatch(emb={emb.shape[0]},blocks={len(blocks)})"

    labels = overlap_labels(blocks, gt_words, threshold, min_words)  # (n,)
    return (emb, labels), source


def iter_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def diagnose_first_row(path):
    """Print what the first row actually contains, so label problems surface early."""
    for row in iter_jsonl(path):
        present = {k: bool(v) for k, v in row.items()}
        _, n_cc = cc_select_words(row.get("html") or "")
        print("first row fields (non-empty):",
              {k: v for k, v in present.items()})
        print(f"first row cc-select nodes: {n_cc}")
        if n_cc == 0:
            print("  WARNING: no cc-select nodes -> will fall back to text fields. "
                  "Check that a content field is populated.")
        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="path to webmainbench.jsonl")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="min fraction of a block's words that must be in the GT vocab")
    ap.add_argument("--min-words", type=int, default=3,
                    help="blocks shorter than this are forced to label 0 (reduces noise)")
    ap.add_argument("--limit", type=int, default=0, help="process only first N rows (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"jsonl={args.jsonl}  out={args.out}  threshold={args.threshold}  min_words={args.min_words}")
    diagnose_first_row(args.jsonl)

    kept = skipped = 0
    sources = {}              # how labels were derived, for the summary
    total_blocks = total_pos = 0

    for i, row in enumerate(iter_jsonl(args.jsonl)):
        if args.limit and kept >= args.limit:
            break

        track_id = row.get("track_id") or f"{i:06d}"
        path = os.path.join(args.out, f"{track_id}.npz")
        if os.path.exists(path):
            continue

        try:
            out, info = process(row, args.threshold, args.min_words)
        except Exception as e:
            print(f"[{i}] skip ({type(e).__name__}): {e}")
            skipped += 1
            continue

        if out is None:
            skipped += 1
            continue

        emb, labels = out
        np.savez(path, emb=emb, labels=labels)
        kept += 1
        sources[info] = sources.get(info, 0) + 1
        total_blocks += len(labels)
        total_pos += int(labels.sum())

        if kept % 200 == 0:
            pos_rate = total_pos / max(total_blocks, 1)
            print(f"[{i}] cached={kept} skipped={skipped}  "
                  f"last(blocks={len(labels)}, pos={int(labels.sum())})  "
                  f"overall pos-rate={pos_rate:.3f}")

    print("-" * 60)
    print(f"done. cached={kept} skipped={skipped}")
    print(f"label sources: {sources}")
    if total_blocks:
        print(f"overall positive block rate: {total_pos / total_blocks:.3f}")
        if total_pos == 0:
            print("  WARNING: zero positive labels -> labeling is broken, do NOT train on this.")


if __name__ == "__main__":
    main()