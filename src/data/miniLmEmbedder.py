"""
minilm_embedder.py
==================
Minimal: simplify_html -> blocks -> all-MiniLM-L6-v2 embeddings (384-dim) -> BiLSTM.
"""

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from mineru_html.process.simplify_html import simplify_html

EMB_DIM = 384

_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def _parse_blocks(html_str):
    """Parse simplified/mapping HTML -> flat ordered list of blocks (elements with _item_id)."""
    soup = BeautifulSoup(html_str, "html.parser")
    blocks = soup.find_all(attrs={"_item_id": True})
    blocks.sort(key=lambda b: int(b.get("_item_id", 0)))
    return blocks

def embed_blocks(blocks):
    """list of parsed blocks -> (len(blocks), 384) embeddings."""
    return _model.encode([str(b) for b in blocks])

def embed_document(raw_html):
    """raw HTML -> (seq_len, 384) embedding matrix, one row per simplified block."""
    simplified_html_str, mapping_html_str = simplify_html(raw_html)
    blocks = _parse_blocks(simplified_html_str)
    return embed_blocks(blocks)