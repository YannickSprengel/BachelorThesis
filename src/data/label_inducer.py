"""
label_inducer.py

The ONLY preprocessing logic you need to write yourself.
Everything else (simplify_html, extract_main_html) comes from mineru_html.

Takes (element_list, clean_text_ground_truth) and returns per-element
binary labels {item_id: 0/1} via token overlap.
"""

import re


def induce_labels(elements, clean_text: str, threshold: float = 0.5) -> dict[int, int]:
    """
    Derive per-element binary labels from ground-truth clean text.

    Args:
        elements:   List of element dicts from mineru_html's simplify_html.
                    Each has keys: item_id, tag, text (or similar — inspect
                    what simplify_html actually returns and adjust below).
        clean_text: Ground-truth plain text for this page (from WebMainBench).
        threshold:  Minimum fraction of element tokens that must appear in
                    clean_text to assign label=1 (main content).

    Returns:
        {item_id: label} where label is 1 (main) or 0 (boilerplate).
    """
    ALWAYS_BOILERPLATE = {"nav", "footer", "header", "aside", "form"}
    MIN_TEXT_LEN = 20  # ignore near-empty elements

    gt_tokens = _tokenize(clean_text)
    labels = {}

    for elem in elements:
        item_id = elem["item_id"]
        tag     = elem.get("tag", "")
        text    = elem.get("text", "").strip()

        # Structural boilerplate: always 0
        if tag in ALWAYS_BOILERPLATE:
            labels[item_id] = 0
            continue

        # Too short to be meaningful
        if len(text) < MIN_TEXT_LEN:
            labels[item_id] = 0
            continue

        # Token overlap with ground truth
        elem_tokens = _tokenize(text)
        if not elem_tokens:
            labels[item_id] = 0
            continue

        overlap = len(elem_tokens & gt_tokens) / len(elem_tokens)
        labels[item_id] = 1 if overlap >= threshold else 0

    return labels


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens as a set (order-independent for overlap)."""
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))
