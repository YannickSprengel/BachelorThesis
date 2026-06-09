"""
Combined per-block representation for the BiLSTM tagger:

    block -> MiniLM embedding (384) + hand-engineered features (47)  =  431 dims

The hand features are already scaled to ~[0,1] in the extractor, and MiniLM values are
small-magnitude too, so the two halves are concatenated directly (no extra normalisation).
"""

import re
import numpy as np
from bs4 import Tag

# reuse the exact simplify + parsing + MiniLM encoding from the MiniLM path
from src.data.miniLmEmbedder import (
    simplify_html,
    _parse_blocks,
    embed_blocks as _minilm_embed,
    EMB_DIM,                       # 384
)

_TAG_VOCAB = [
    'div', 'p', 'span', 'a',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'li', 'ul', 'ol',
    'table', 'td', 'th', 'tr',
    'article', 'section', 'main', 'aside',
    'blockquote', 'pre', 'code',
    'figure', 'figcaption', 'img',
    'other',
]
_CONTENT_KEYWORDS = ['article', 'content', 'main', 'post', 'body', 'text',
                     'story', 'blog', 'entry', 'detail', 'description']
_BOILERPLATE_KW = ['nav', 'menu', 'footer', 'header', 'sidebar', 'ad', 'ads',
                   'banner', 'social', 'share', 'comment', 'related',
                   'recommend', 'cookie', 'popup', 'breadcrumb', 'pagination']

FEATURE_DIM = len(_TAG_VOCAB) + 8 + 2 + 2 + 1 + 2 + 4   # = 47
COMBINED_DIM = EMB_DIM + FEATURE_DIM                    # = 431


def _max_depth(tag, depth=0):
    children = [c for c in tag.children if isinstance(c, Tag)]
    if not children:
        return depth
    return max(_max_depth(c, depth + 1) for c in children)


def hand_features(block, block_idx, total_blocks):
    """47-dim structural feature vector for one block (verbatim from BlockFeatureExtractor)."""
    f = np.zeros(FEATURE_DIM, dtype=np.float32)
    ptr = 0

    # tag type one-hot (28)
    tag = (block.name or 'other').lower()
    idx = _TAG_VOCAB.index(tag) if tag in _TAG_VOCAB else _TAG_VOCAB.index('other')
    f[ptr + idx] = 1.0
    ptr += len(_TAG_VOCAB)

    # text statistics (8)
    text = block.get_text(separator=' ', strip=True)
    words = text.split()
    wc, cc = len(words), len(text)
    f[ptr + 0] = min(cc / 1000.0, 1.0)
    f[ptr + 1] = min(wc / 200.0, 1.0)
    f[ptr + 2] = len(re.findall(r'[.!?]', text)) / max(wc, 1)
    f[ptr + 3] = sum(1 for c in text if c.isupper()) / max(cc, 1)
    f[ptr + 4] = (float(np.mean([len(w) for w in words])) / 10.0) if words else 0.0
    f[ptr + 5] = len(re.findall(r'\d', text)) / max(cc, 1)
    f[ptr + 6] = len(re.findall(r'[,;:]', text)) / max(wc, 1)
    f[ptr + 7] = min(len(re.findall(r'\n', text)) / 10.0, 1.0)
    ptr += 8

    # link features (2)
    anchors = block.find_all('a')
    f[ptr + 0] = sum(len(a.get_text()) for a in anchors) / max(cc, 1)
    f[ptr + 1] = min(len(anchors) / 10.0, 1.0)
    ptr += 2

    # DOM nesting (2)
    f[ptr + 0] = min(len(block.find_all(True)) / 30.0, 1.0)
    f[ptr + 1] = min(_max_depth(block) / 8.0, 1.0)
    ptr += 2

    # document position (1)
    f[ptr] = block_idx / max(total_blocks - 1, 1)
    ptr += 1

    # class/id keyword scores (2)
    cid = ' '.join([' '.join(block.get('class') or []), block.get('id') or '']).lower()
    f[ptr + 0] = min(sum(kw in cid for kw in _CONTENT_KEYWORDS) / 3.0, 1.0)
    f[ptr + 1] = min(sum(kw in cid for kw in _BOILERPLATE_KW) / 3.0, 1.0)
    ptr += 2

    # binary child-tag flags (4)
    f[ptr + 0] = 1.0 if block.find('table') else 0.0
    f[ptr + 1] = 1.0 if block.find(['code', 'pre']) else 0.0
    f[ptr + 2] = 1.0 if block.find('img') else 0.0
    f[ptr + 3] = 1.0 if block.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']) else 0.0
    return f


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------
def embed_blocks(blocks):
    """list of parsed blocks -> (len(blocks), 431) = [MiniLM 384 | hand 47]."""
    n = len(blocks)
    if n == 0:
        return np.zeros((0, COMBINED_DIM), dtype=np.float32)
    sem = np.asarray(_minilm_embed(blocks), dtype=np.float32)               # (n, 384)
    hand = np.stack([hand_features(b, i, n) for i, b in enumerate(blocks)])  # (n, 47)
    return np.concatenate([sem, hand], axis=1)                              # (n, 431)


def embed_document(raw_html):
    """raw HTML -> (seq_len, 431) combined embedding matrix, one row per simplified block."""
    simplified_html_str, _ = simplify_html(raw_html)
    return embed_blocks(_parse_blocks(simplified_html_str))


if __name__ == "__main__":
    print("EMB_DIM =", EMB_DIM, " FEATURE_DIM =", FEATURE_DIM, " COMBINED_DIM =", COMBINED_DIM)
    demo = "<html><body><article _item_id='0' class='post-content'><h1>Hi</h1>" \
           "<p>Some real text here with words.</p></article>" \
           "<nav _item_id='1' class='menu'><a href='#'>home</a></nav></body></html>"
    out = embed_document(demo)
    print("embed_document shape:", out.shape)   # expect (2, 431)