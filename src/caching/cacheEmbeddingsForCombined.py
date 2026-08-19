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


# Ground-truth vocabulary
def cc_select_words(html):
    """Set of words inside all cc-select="true" regions of the original html.
    """
    soup = BeautifulSoup(html, "html.parser")
    nodes = [n for n in soup.find_all(attrs={"cc-select": True})
             if str(n.get("cc-select")).lower() == "true"]
    words = set()
    for n in nodes:
        words.update(tokenize(n.get_text(" ", strip=True)))
    return words, len(nodes)


def fallback_words(row):
    """Ground-truth vocabulary from convert_main_content / groundtruth_content /
    main_html, in that order -- used when a page has no usable cc-select markup at
    all, by both the default (--labeler overlap) and --labeler dom code paths."""
    for field in ("convert_main_content", "groundtruth_content"):
        txt = row.get(field)
        if txt:
            return set(tokenize(txt)), field

    main_html = row.get("main_html")
    if main_html:
        text = BeautifulSoup(main_html, "html.parser").get_text(" ", strip=True)
        return set(tokenize(text)), "main_html"

    return set(), None


def groundtruth_words(row):
    words, n = cc_select_words(row.get("html") or "")
    if n > 0 and words:
        return words, "cc-select"
    return fallback_words(row)


def dom_correspondence_labels(simpl_blocks, mapping_blocks):
    """Position-precise alternative to word-overlap: simplify_html's process_paragraphs
    already propagates cc-select="true" onto whichever mapping-tree node ends up
    carrying the matching _item_id (both for genuine block-level elements and for
    synthetic cc-alg-uc-text wrappers around loose text runs) -- see
    analysis/oracle_investigation.md for how this was found and measured. So the label
    is a direct attribute read, not a heuristic: no word-overlap computation needed.

    A block whose _item_id has no counterpart in the mapping tree at all (real,
    measured possibility -- process_paragraphs's non-block-element wrapping logic can
    silently fail to find an exact-text match after upstream truncation/whitespace
    normalization) is labeled 0 but counted separately as "unmatched", not silently
    folded into the boilerplate label with no visibility.

    Returns (labels, n_marked, n_unmatched)."""
    map_by_id = {b.get("_item_id"): b for b in mapping_blocks}
    labels = np.zeros(len(simpl_blocks), dtype=np.float32)
    n_marked = n_unmatched = 0
    for i, b in enumerate(simpl_blocks):
        mb = map_by_id.get(b.get("_item_id"))
        if mb is None:
            n_unmatched += 1
            continue
        if str(mb.get("cc-select", "")).lower() == "true":
            labels[i] = 1.0
            n_marked += 1
    return labels, n_marked, n_unmatched


def overlap_labels(blocks, gt_words, threshold, min_words):
    """1 if >= threshold of a block's words appear in the ground-truth vocabulary.
    A block under min_words needs a full match (all its words present) instead of just
    threshold: a partial match on that few words is too easily coincidence, but a full
    match is trusted even for a one-word block (e.g. a section header like "ARTS ...")."""
    labels = np.zeros(len(blocks), dtype=np.float32)
    for i, b in enumerate(blocks):
        w = block_words(b)
        if not w:
            continue
        frac = sum(t in gt_words for t in w) / len(w)
        needed = threshold if len(w) >= min_words else 1.0
        if frac >= needed:
            labels[i] = 1.0
    return labels


# Per-document processing
def process(row, threshold, min_words, labeler="overlap"):
    """labeler="overlap": today's word-overlap heuristic, unchanged.
    labeler="dom": direct cc-select attribute read via dom_correspondence_labels,
    falling back to word-overlap (against convert_main_content/groundtruth_content/
    main_html only, NOT cc-select -- already tried and found nothing marked) for
    pages with no usable cc-select markup at all. Returns ((emb, labels, n_unmatched), source)."""
    html = row.get("html")
    if not html:
        return None, "no-html"

    simplified_html_str, mapping_html_str = simplify_html(html)
    blocks = _parse_blocks(simplified_html_str)
    if not blocks:
        return None, "no-blocks"

    n_unmatched = 0
    if labeler == "dom":
        mapping_blocks = _parse_blocks(mapping_html_str)
        labels, n_marked, n_unmatched = dom_correspondence_labels(blocks, mapping_blocks)
        if n_marked > 0:
            source = "cc-select-dom"
        else:
            _n_regions_raw = cc_select_words(html)[1]
            gt_words, source = fallback_words(row)
            if not gt_words:
                return None, "no-label-source"
            source = f"dom-fallback:{source}" if _n_regions_raw == 0 else f"dom-fallback-propagation-empty:{source}"
            labels = overlap_labels(blocks, gt_words, threshold, min_words)
    else:
        gt_words, source = groundtruth_words(row)
        if not gt_words:
            return None, "no-label-source"
        labels = overlap_labels(blocks, gt_words, threshold, min_words)

    emb = np.asarray(embed_document(html), dtype=np.float32)   # (n, 384)
    if emb.shape[0] != len(blocks):
        return None, f"align-mismatch(emb={emb.shape[0]},blocks={len(blocks)})"

    return (emb, labels, n_unmatched), source


def iter_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[jsonl:{lineno}] skip malformed line: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="path to webmainbench.jsonl")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--labeler", choices=["overlap", "dom"], default="overlap",
                    help="overlap: word-overlap against cc-select vocabulary (default, "
                         "unchanged legacy behavior). dom: direct cc-select attribute read "
                         "via the mapping tree, see analysis/oracle_investigation.md")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="min fraction of a block's words that must be in the GT vocab")
    ap.add_argument("--min-words", type=int, default=3,
                    help="blocks shorter than this need a full word match, not just threshold")
    ap.add_argument("--limit", type=int, default=0, help="process only first N rows (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"jsonl={args.jsonl}  out={args.out}  labeler={args.labeler}  "
          f"threshold={args.threshold}  min_words={args.min_words}")

    kept = skipped = 0
    sources = {}              # how labels were derived, for the summary
    total_blocks = total_pos = total_unmatched = 0

    for i, row in enumerate(iter_jsonl(args.jsonl)):
        if args.limit and kept >= args.limit:
            break

        track_id = row.get("track_id") or f"{i:06d}"
        path = os.path.join(args.out, f"{track_id}.npz")
        if os.path.exists(path):
            continue

        try:
            out, info = process(row, args.threshold, args.min_words, args.labeler)
        except Exception as e:
            print(f"[{i}] skip ({type(e).__name__}): {e}")
            skipped += 1
            continue

        if out is None:
            skipped += 1
            continue

        emb, labels, n_unmatched = out
        np.savez(path, emb=emb, labels=labels, label_method=info)
        kept += 1
        sources[info] = sources.get(info, 0) + 1
        total_blocks += len(labels)
        total_pos += int(labels.sum())
        total_unmatched += n_unmatched

        if kept % 200 == 0:
            pos_rate = total_pos / max(total_blocks, 1)
            unmatched_rate = total_unmatched / max(total_blocks, 1)
            print(f"[{i}] cached={kept} skipped={skipped}  "
                  f"last(blocks={len(labels)}, pos={int(labels.sum())})  "
                  f"overall pos-rate={pos_rate:.3f}  unmatched-rate={unmatched_rate:.4f}")

    print("-" * 60)
    print(f"done. cached={kept} skipped={skipped}")
    print(f"label sources: {sources}")
    if total_blocks:
        print(f"overall positive block rate: {total_pos / total_blocks:.3f}")
        if total_pos == 0:
            print("  WARNING: zero positive labels -> labeling is broken, do NOT train on this.")
        if args.labeler == "dom":
            unmatched_rate = total_unmatched / total_blocks
            print(f"overall unmatched-block rate (dom labeler): {unmatched_rate:.4f} "
                  f"({total_unmatched}/{total_blocks})")
            if unmatched_rate > 0.02:
                print("  WARNING: >2% of blocks have no mapping-tree counterpart -- "
                      "labels for those blocks default to 0 with no real signal behind "
                      "them. Investigate before training on this cache.")


if __name__ == "__main__":
    main()