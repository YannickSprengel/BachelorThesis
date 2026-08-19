"""
Block-level plumbing shared by every WCEB check: a trained model's evaluation and the
model-free oracle ceiling check both parse pages into blocks, decide keep/drop, and
reconstruct the kept content back into HTML the same way.
"""

import re
from collections import Counter
from bs4 import BeautifulSoup
from mineru_html.process.simplify_html import simplify_html
from src.evaluation.textMetrics import tokenize, _lcs_with_endpoint

_HAS_WORD_CHAR = re.compile(r"\w")


def _parse_blocks(html_str):
    """Parse simplified/mapping HTML -> flat ordered list of blocks (elements with _item_id).
    Duplicated from src.data.miniLmEmbedder on purpose: this module must not import
    anything that loads the MiniLM model, so the oracle ceiling check stays model-free."""
    soup = BeautifulSoup(html_str, "html.parser")
    blocks = soup.find_all(attrs={"_item_id": True})
    blocks.sort(key=lambda b: int(b.get("_item_id", 0)))
    return blocks


def parse_page(html):
    """raw html -> (simpl_blocks, mapping_blocks).
    simpl_blocks: cleaned/truncated view. What a model embeds and is scored against.
    mapping_blocks: full untruncated original DOM, matched to simpl_blocks by
    _item_id. Used only to reconstruct the final output content."""
    simp_str, map_str = simplify_html(html)
    return _parse_blocks(simp_str), _parse_blocks(map_str)


def block_texts(blocks):
    return [b.get_text(" ", strip=True) for b in blocks]


def reconstruct(simpl_blocks, mapping_blocks, keep):
    """keep: bool/int sequence aligned 1:1 with simpl_blocks. Pulls the corresponding
    mapping block (full untruncated content) for every kept block, then drops a kept
    block if one of its ancestors is also kept, to avoid duplicated nested content."""
    map_by_id = {b.get("_item_id"): b for b in mapping_blocks}
    kept = [map_by_id[b.get("_item_id")]
            for b, k in zip(simpl_blocks, keep)
            if k and b.get("_item_id") in map_by_id]

    kept_ids = {id(b) for b in kept}
    top = [b for b in kept if not any(id(a) in kept_ids for a in b.parents)]
    return "\n".join(str(b) for b in top)


def _words(text):
    """Lowercased word tokens, punctuation and whitespace dropped. Uses the same
    jieba-based tokenizer as the ROUGE metrics so a script without spaces between
    words (Chinese, Japanese) is split at word granularity instead of \\w+ grabbing
    a whole punctuation-delimited clause as a single "word"."""
    return [t.lower() for t in tokenize(text) if _HAS_WORD_CHAR.search(t)]


def overlap_labels(texts, gt, threshold=0.5, min_words=3):
    """1 if >= threshold of a block's tokens appear in the ground-truth text, else 0.
    Same heuristic used to build training labels from WebMainBench's cc-select regions.
    A block under min_words needs a full match (all its tokens present) instead of just
    threshold: a partial match on that few tokens is too easily coincidence, but a full
    match is trusted even for a one-word block (e.g. a section header like "ARTS ...")."""
    gt_tokens = set(_words(gt or ""))
    out = []
    for text in texts:
        toks = _words(text or "")
        if not toks:
            out.append(0)
            continue
        frac = sum(1 for t in toks if t in gt_tokens) / len(toks)
        needed = threshold if len(toks) >= min_words else 1.0
        out.append(1 if frac >= needed else 0)
    return out


def overlap_labels_weighted(texts, gt, threshold=0.5, min_words=3):
    """Frequency-aware variant of overlap_labels: gt_tokens is a Counter, not a set, and
    a block's matched count is clipped multiset overlap (same Counter & Counter clipping
    rouge_n_f1 itself uses), not set membership. A block that repeats one rare gt word
    many times (e.g. a nav teaser reusing the headline) no longer gets free credit for
    every repetition -- each occurrence in gt can only be matched once."""
    gt_counts = Counter(_words(gt or ""))
    out = []
    for text in texts:
        toks = _words(text or "")
        if not toks:
            out.append(0)
            continue
        matched = sum((Counter(toks) & gt_counts).values())
        frac = matched / len(toks)
        needed = threshold if len(toks) >= min_words else 1.0
        out.append(1 if frac >= needed else 0)
    return out


def overlap_labels_sequential(texts, gt, threshold=0.5, min_words=3):
    """Position-aware variant of overlap_labels: instead of checking whether a block's
    words appear ANYWHERE in gt (order/frequency-blind, see overlap_labels), search for
    the block's best alignment in a forward-moving window of gt starting where the
    previous ACCEPTED block's match left off. Blocks are already in document order
    (parse_page sorts by _item_id), and genuine content preserves that order in gt too
    -- validated empirically across all 8 WCEB sub-datasets, see
    analysis/oracle_investigation.md Part 5. A rejected block leaves the cursor where it
    was, so the next block still searches from the same point.

    A widened-window retry was tried and dropped during development: it let a
    low-quality near-miss (e.g. an author byline that isn't really present in gt)
    accept via a spurious, widely-scattered match far ahead in the token stream,
    over-advancing the cursor and stranding a genuine later block behind it (a real,
    observed regression, not a hypothetical one -- see analysis/oracle_investigation.md
    Part 5). The SPAN_GUARD_MULT check below is therefore an accept/reject gate, not
    just an advance-amount decision: a match whose span is disproportionate to the
    block's own length is weak evidence and gets rejected outright, on the theory that
    a genuine positional match should be reasonably tight, not just technically present
    somewhere in a wide window.

    Tuning constants below (window floor/multiplier/cap, span-guard multiplier) are
    internal, not exposed via the shared (texts, gt, threshold, min_words) labeler
    signature every overlap_labels* variant uses -- see the analysis doc for how they
    were chosen and what each guards against.
    """
    WINDOW_FLOOR = 100
    WINDOW_MULT = 6
    WINDOW_CAP = 1000
    SPAN_GUARD_MULT = 3

    gt_tokens = _words(gt or "")
    cursor = 0
    out = []
    for text in texts:
        toks = _words(text or "")
        if not toks:
            out.append(0)
            continue

        is_short = len(toks) < min_words
        needed = 1.0 if is_short else threshold
        remaining = max(0, len(gt_tokens) - cursor)
        if is_short:
            # no floor for short blocks -- a 100-token floor would make a 3-word
            # block's "local" window basically global again, reintroducing the exact
            # coincidental-match risk this whole approach is meant to avoid.
            size = min(remaining, WINDOW_MULT * len(toks))
        else:
            size = min(remaining, max(WINDOW_FLOOR, WINDOW_MULT * len(toks)), WINDOW_CAP)

        window = gt_tokens[cursor: cursor + size]
        length, first, last = _lcs_with_endpoint(toks, window)
        frac = length / len(toks)
        span = (last - first + 1) if last is not None else 0
        loose = span > SPAN_GUARD_MULT * len(toks)

        if is_short:
            # near-contiguity required, not just "matched somewhere in the window" --
            # a short phrase spread thinly across a wide span is weak evidence even at
            # 100% token coverage (e.g. "Best of Toronto" against a 2500-token article
            # that happens to use "Toronto" once, far from "Best"/"of").
            accept = frac >= needed and span <= len(toks) + 2
        else:
            accept = frac >= needed and not loose

        if accept:
            out.append(1)
            cursor += last + 1
        else:
            out.append(0)
            # cursor intentionally left unchanged: a wrongly-rejected block becomes an
            # isolated false negative, it doesn't corrupt later blocks' own searches.
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f
