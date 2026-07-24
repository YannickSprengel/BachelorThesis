"""
Block-level plumbing shared by every WCEB check: a trained model's evaluation and the
model-free oracle ceiling check both parse pages into blocks, decide keep/drop, and
reconstruct the kept content back into HTML the same way.
"""

import re
from bs4 import BeautifulSoup
from mineru_html.process.simplify_html import simplify_html
from src.evaluation.textMetrics import tokenize

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


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f
