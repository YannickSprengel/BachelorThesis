"""
Text-similarity metrics for comparing extracted plaintext against WCEB ground truth.
Both metrics tokenize with jieba, matching what Dripper used for WCEB.
"""

from collections import Counter
import jieba


def _tokenize(text):
    return jieba.lcut(text or "")


def rouge_n_f1(pred, ref, n=5):
    """ROUGE-N F1: overlap of contiguous n-grams. Needs n identical tokens in a row to
    count as a match, so one wrong/missing/extra token breaks every n-gram window
    touching it."""
    pt, rt = _tokenize(pred), _tokenize(ref)
    ng = lambda t: Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
    pg, rg = ng(pt), ng(rt)
    if not pg or not rg:
        return 0.0
    overlap = sum((pg & rg).values())          # clipped n-gram overlap
    if overlap == 0:
        return 0.0
    prec, rec = overlap / sum(pg.values()), overlap / sum(rg.values())
    return 2 * prec * rec / (prec + rec)


def _lcs_length(a, b):
    """Length of the longest common subsequence of two token lists.
    O(len(a)*len(b)) time, O(min(len(a),len(b))) space."""
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l_f1(pred, ref):
    """ROUGE-L F1: LCS-based. Matched tokens don't need to be contiguous, just in the
    same relative order, so this tolerates insertions/deletions/reordering much better
    than rouge_n_f1 does."""
    pt, rt = _tokenize(pred), _tokenize(ref)
    if not pt or not rt:
        return 0.0
    lcs = _lcs_length(pt, rt)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(pt), lcs / len(rt)
    return 2 * prec * rec / (prec + rec)


def to_text(html):
    """HTML -> plain text. Uses html-text (what Dripper used for WCEB); falls back to bs4."""
    try:
        import html_text
        return html_text.extract_text(html or "")
    except ImportError:
        from bs4 import BeautifulSoup
        return BeautifulSoup(html or "", "html.parser").get_text(" ", strip=True)
