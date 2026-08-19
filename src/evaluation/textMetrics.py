"""
Text-similarity metrics for comparing extracted plaintext against WCEB ground truth.
Both metrics tokenize with jieba, matching what Dripper used for WCEB.
"""

import re
from collections import Counter
import jieba

_DASHES = re.compile(r"[‒–—―]")   # figure/en/em/horizontal-bar dashes
_SINGLE_QUOTES = re.compile(r"[‘’‚‛]")
_DOUBLE_QUOTES = re.compile(r"[“”„‟]")

# jieba only recognizes plain ASCII as "non-Chinese"; an accented Latin word like
# "Regístrate" gets shattered into fragments ('Reg', 'í', 'strate') because it treats
# the accented letter as a lone CJK-like character. Route actual CJK text (Han,
# Hiragana/Katakana, Hangul) through jieba and everything else through a plain
# word-boundary split, so accented Latin scripts stay intact.
_CJK_RUN = re.compile(r"[一-鿿぀-ヿ가-힯]+")
_WORD = re.compile(r"\w+")


def _normalize_typography(text):
    """Collapse typographic punctuation to the ASCII form ground-truth sources use
    inconsistently (e.g. cleaneval/l3s-gn1 write em-dashes as "--", cetd/dragnet keep
    the real unicode char). Applied to both pred and ref so comparisons aren't thrown
    off by which convention a given source happened to use."""
    text = _DASHES.sub("--", text)
    text = _SINGLE_QUOTES.sub("'", text)
    text = _DOUBLE_QUOTES.sub('"', text)
    text = text.replace("…", "...")
    return text


def tokenize(text):
    """Word list for a string. CJK runs go through jieba for word-granularity
    segmentation (a run of Chinese/Japanese characters has no spaces to split on);
    everything else (including accented Latin scripts) is split on \\w+ boundaries,
    since jieba mishandles non-ASCII Latin text. Shared with
    blockReconstruction.overlap_labels so labeling and scoring agree on what a
    "word" is."""
    text = _normalize_typography(text or "")
    tokens = []
    pos = 0
    for m in _CJK_RUN.finditer(text):
        tokens.extend(_WORD.findall(text[pos:m.start()]))
        tokens.extend(jieba.lcut(m.group()))
        pos = m.end()
    tokens.extend(_WORD.findall(text[pos:]))
    return tokens


def rouge_n_f1(pred, ref, n=5):
    """ROUGE-N F1: overlap of contiguous n-grams. Needs n identical tokens in a row to
    count as a match, so one wrong/missing/extra token breaks every n-gram window
    touching it."""
    pt, rt = tokenize(pred), tokenize(ref)
    if not pt and not rt:
        return 1.0  # nothing to extract, nothing extracted: correct
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


def _lcs_with_endpoint(a, b):
    """LCS of a vs b, PLUS the (first, last) index range in b spanned by the chosen
    alignment -- used by blockReconstruction.overlap_labels_sequential to know where in
    the ground-truth token stream a block's match sits, not just how long it is.

    Deliberately does NOT use _lcs_length's swap-for-space trick (swapping a/b so the
    rolling array tracks the shorter side): callers need indices into b specifically
    (b is always the ground-truth window here), and swapping would silently flip which
    sequence the returned indices refer to. b is bounded by the caller's window cap, so
    the extra space this costs is negligible.

    On a length tie between two ways to extend the alignment, keeps the one with the
    SMALLER last-index (minimal-consumption alignment) -- this is the endpoint contract
    callers rely on: advancing a search cursor to the smallest position that still
    justifies the match, not an arbitrary maximal-reach one.

    Returns (length, first_index_in_b, last_index_in_b); first/last are None if length==0.
    O(len(a)*len(b)) time, O(len(b)) space.
    """
    if not a or not b:
        return 0, None, None

    def better(c1, c2):
        """Prefer higher length; on a tie, prefer the smaller last-index (and treat a
        0-length cell, which has no meaningful endpoint, as ties among themselves)."""
        if c1[0] != c2[0]:
            return c1 if c1[0] > c2[0] else c2
        if c1[0] == 0:
            return c1
        return c1 if c1[2] <= c2[2] else c2

    NONE_CELL = (0, None, None)
    prev = [NONE_CELL] * (len(b) + 1)
    for x in a:
        curr = [NONE_CELL] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                diag_len, diag_first, _diag_last = prev[j - 1]
                first = diag_first if diag_first is not None else j - 1
                # "up" (prev[j]) is provably dominated whenever a match is available
                # (extending the window by one token can raise the LCS by at most 1),
                # so only "left" (curr[j-1]) can tie/beat the fresh diagonal match --
                # e.g. the same phrase occurring twice in b: without this comparison,
                # the LATER occurrence would silently overwrite an equal-length match
                # already propagated from the EARLIER one.
                curr[j] = better((diag_len + 1, first, j - 1), curr[j - 1])
            else:
                curr[j] = better(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l_f1(pred, ref):
    """ROUGE-L F1: LCS-based. Matched tokens don't need to be contiguous, just in the
    same relative order, so this tolerates insertions/deletions/reordering much better
    than rouge_n_f1 does."""
    pt, rt = tokenize(pred), tokenize(ref)
    if not pt and not rt:
        return 1.0  # nothing to extract, nothing extracted: correct
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
