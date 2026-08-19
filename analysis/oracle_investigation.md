# Oracle-ceiling investigation

Why does the WCEB oracle ceiling (gold token-overlap labels, no model — `evaluateOracleCeiling.py`)
sit at ROUGE-5 F1 = 0.9013 instead of near 1.0, and why do GRU/xLSTM already meet-or-beat it on
some WCEB sub-datasets (`overview.md` Finding 3) while trailing it on others? Source data:
`results/oracle/wceb.{csv,json}`, `results/oracle_diagnosis/*.csv`,
`results/trueCeiling/wceb_worst.{csv,json}`, `analysis/stats/bare_cell_hypothesis.{md,json}`,
`analysis/stats/oracle_vs_oracle_weighted.{md,json}`. Every number below traces to one of those
files.

## Determinism: checked and ruled out

The question was raised whether block generation (`mineru_html.process.simplify_html`) is fully
deterministic — if the same HTML could segment into different blocks across runs, that alone could
explain instability. Read through the vendored 1172-line implementation this session:
`_item_id` assignment is a deterministic sequential DFS walk over the DOM (no dependency on dict/set
iteration order, hashing, or wall-clock time). The only randomness in the file is a `uuid.uuid4()`
temporary `data-uid` attribute used purely to match nodes between a deep-copied tree and the
original during processing — it's unconditionally stripped before `simplify_html()` returns and
never influences `_item_id` order, block segmentation, or leaks into output text/attributes. **Block
generation is not the bug.** The real drivers are elsewhere, below.

## What actually caps the ceiling: the silver-label heuristic, not the pipeline

`evaluateOracleCeiling.py` uses `overlap_labels()` (`blockReconstruction.py:60-76`) to decide
keep/drop per block: `gt_tokens = set(_words(gt))` — a block passes if ≥50% of its own tokens
appear *anywhere* in the ground truth's word bag. This is **order- and frequency-blind**: it has no
concept of position, contiguity, or how many times a word repeats. ROUGE-5, the metric it's
supposed to be optimizing for, needs 5 contiguous, position-respecting tokens to count a match — a
completely different objective. This heuristic is also literally the training-label heuristic (same
function feeds `cacheEmbeddings*.py`), so a flaw here caps the oracle *and* is the signal every
tagger is trained against.

### The mean is tail-driven, not a uniform shortfall

Oracle mean ROUGE-5 = 0.9013, but **median = 0.9301**. 43 of 3985 pages score below 0.5, mostly in
dragnet (16), cleaneval (10), google-trends-2017 (7), l3s-gn1 (7) — a small, tractable "worst tail"
responsible for most of the mean-vs-median gap.

## Part 1: how much headroom exists at this exact block granularity? (greedy toggle search)

New script `evaluateTrueCeiling.py`: starting from the oracle's own keep mask, a coordinate-ascent
search repeatedly flips whichever single block improves the page's ROUGE-5 the most, until a full
pass makes no more flips. This measures the **true ceiling achievable at the same block
segmentation/reconstruction**, independent of the labeling heuristic — a *conservative lower bound*
(single-flip search can't find improvements needing two blocks toggled jointly).

Run on the 43-page worst tail (2 pages skipped, >2500 blocks each, logged not dropped):

| dataset | mean gain over oracle (n pages) |
|---|---|
| dragnet | **+0.575** |
| l3s-gn1 | +0.508 |
| google-trends-2017 | +0.348 |
| cetd | +0.218 |
| readability | +0.156 |
| cleaneval | +0.095 |
| **overall** | **+0.390** (n=41) |

This is a large, unambiguous result. Individual dragnet pages jump from oracle scores of 0.21–0.47
all the way to 0.95–1.00 true ceiling — e.g. `dragnet/3e502bbb`: oracle 0.2087 → true ceiling
0.9884 (converged in 20 flips). **On the pages where the oracle does worst, the silver-label
heuristic is picking a badly wrong block set, and a much better set exists at the identical
granularity.** Two pages stayed at 0.0 with zero flips found (`l3s-gn1/b78d17d3`, an oracle
complete-miss with `n_kept=0`; `readability/e6f0b278`, a ROUGE-N zero-token-cliff artifact — its
`rouge_l=1.0` confirms the content is actually perfectly recovered, it's just too short for a
5-gram, see below) — single-flip search cannot rescue either failure mode.

## Part 2: worst-tail forensics — what's actually going wrong (manual inspection, 6 pages)

`diagnoseOraclePages.py --pages worst --dump-diff` dumped token-level diffs (predicted vs. ground
truth) for one representative page per dataset in the worst tail. Two distinct failure modes,
roughly evenly split across the sample:

**Over-inclusion (boilerplate pollution)** — cetd, dragnet, google-trends-2017 (the same three
datasets with the largest true-ceiling headroom above): the reconstruction adds *thousands* of
extra tokens the ground truth doesn't contain (+2055, +2888, +2469 added lines respectively, ~0
deletions). Reading the actual diffs: `dragnet/3e502bbb` pulls in a full site nav menu ("Best of
Toronto People Fashion Style...") and an entire reader-comments section ("Add a Comment / Website /
Comments / Remember me...") because those sections share vocabulary with the article (category
names like "Fashion"/"Style" that the article itself also uses). `google-trends-2017/c0a2dd2a`
similarly pulls in a "Members / Already a subscriber / Sign In" paywall prompt and a list of
unrelated headline teasers. Feature-checked: this page kept 228 of 426 blocks (54%!) with a
`bare_cell_frac` of only 0.9% — this is not a structural/table-cell problem, it's the bag-of-words
heuristic simply keeping about half the page.

**Under-inclusion (real content dropped or missed entirely)** — cleaneval, l3s-gn1: the opposite
pattern, hundreds of deletions and almost no additions (cleaneval: +65/-594; l3s-gn1: +1/-188).
`l3s-gn1/b78d17d3` is a complete miss: `n_kept=0`, the heuristic found no block clearing the 50%
overlap threshold at all, so the reconstruction is empty and the true-ceiling search (Part 1) can't
recover it either (no single block alone reaches 50% overlap; would need a lower threshold or a
different metric to fix, not more search).

**Metric artifact, not a pipeline bug**: `readability/e6f0b278` has an empty diff (predicted and
ground-truth token sequences are literally identical) yet `rouge5=0.0` — a 39-char/3-block page,
too short to contain any 5-gram, so `rouge_n_f1`'s `if not pg or not rg: return 0.0` cliff fires
despite `rouge_l=1.0` confirming perfect content recovery. Checked corpus-wide earlier in this
investigation: only 3/3985 pages hit this exactly, not a broad driver — but it can and does show up
in a small worst-tail sample, worth excluding from "true content-selection quality" framing when it
appears.

## Part 3: bare-cell structural hypothesis — real, but a secondary effect

Confirmed in `mineru_html.process.simplify_html`: for "layout" (non-data) tables/lists, individual
`td`/`th`/`li`/`dt`/`dd` cells become standalone blocks with **no** `<table>`/`<tr>`/`<ul>` wrapper
(`extract_paragraphs(..., include_parents=False)`), and `reconstruct()` joins whatever's kept with a
flat `"\n"` — no row/column/bullet structure. Tested across all 3985 pages, per dataset (controls
for dragnet's known higher table-density as a confound):

| dataset | prevalence (has bare cell) | mean rouge5, has vs. no | diff | significant |
|---|---|---|---|---|
| google-trends-2017 | 45.0% | 0.815 vs 0.870 | **−0.055** | yes |
| cetd | 65.1% | 0.910 vs 0.936 | −0.025 | yes |
| dragnet | 43.1% | 0.866 vs 0.888 | −0.022 | yes |
| cleaneval | 54.1% | 0.917 vs 0.925 | −0.008 | no |
| readability | 33.9% | 0.913 vs 0.918 | −0.005 | no |
| scrapinghub | 28.7% | 0.903 vs 0.907 | −0.004 | no |
| l3s-gn1 | 46.5% | 0.920 vs 0.918 | +0.002 | no |
| cleanportaleval | 69.0% | 0.910 vs 0.906 | +0.003 | no |
| **overall** | 49.2% | 0.896 vs 0.907 | **−0.011** | yes |

Bare-cell fragmentation is **common** (49% of all pages have at least one kept bare cell) and its
effect is statistically real in exactly the three datasets with the biggest true-ceiling headroom
(google-trends-2017, cetd, dragnet) — but the effect size (−0.005 to −0.055) is an order of
magnitude smaller than Part 1's true-ceiling gap (+0.39 mean on the worst tail). **Verdict: real,
worth fixing eventually, but not the dominant driver** — the order/frequency-blind labeling
heuristic (Part 1/2) explains far more of the gap.

## Part 4: does a cheap labeling fix help? (frequency-weighted overlap, full 3985-page A/B)

Added `overlap_labels_weighted()` (`blockReconstruction.py`): same threshold logic, but
`gt_tokens` is a `Counter`, and a block's matched count is **clipped multiset overlap**
(`Counter & Counter`, the same clipping `rouge_n_f1` itself uses) instead of set membership — a
block that repeats one rare ground-truth word many times no longer gets free credit for every
repetition. This targets Part 2's boilerplate-pollution failure mode directly (repeated nav/teaser
vocabulary), though it cannot catch pollution made of many *distinct* common words, only repeated
ones — expected to be a partial, not complete, fix.

Full 3985-page run, compared via `compareArchs.py` (`analysis/stats/oracle_vs_oracle_weighted.md`):

**Overall**: ROUGE-5 0.9013 → **0.9047**, mean diff **+0.0035**, 95% CI **[+0.0028, +0.0041] —
significant**. Significant on 7/8 datasets (all but cleaneval, CI [−0.0010, +0.0027]):

| dataset | diff | 95% CI | significant |
|---|---|---|---|
| google-trends-2017 | **+0.0065** | [+0.0034, +0.0111] | yes |
| dragnet | **+0.0051** | [+0.0042, +0.0062] | yes |
| cleanportaleval | +0.0041 | [+0.0008, +0.0087] | yes |
| scrapinghub | +0.0031 | [+0.0012, +0.0055] | yes |
| cetd | +0.0030 | [+0.0023, +0.0038] | yes |
| l3s-gn1 | +0.0025 | [+0.0004, +0.0041] | yes |
| readability | +0.0025 | [+0.0009, +0.0045] | yes |
| cleaneval | +0.0011 | [−0.0010, +0.0027] | no |

**Verdict**: a real, confirmed, statistically significant fix — and it correctly lands hardest on
dragnet and google-trends-2017, the same two datasets Part 1 found the most true-ceiling headroom
in, and the two datasets Part 2 identified as boilerplate-pollution cases. That directional
agreement is a good sanity check: the fix works on the right mechanism. But the *magnitude* is
small — +0.0035 overall recovers only about **1% of the +0.39 mean headroom** the worst tail showed
in Part 1. As expected going in: clipping *repeated* rare-word matches doesn't stop a block full of
many *distinct* common words (a nav menu's category names, a comment section's varied usernames)
from still passing the 50% overlap threshold — the fix addresses one specific failure mode
(repetition-driven false positives) but not the broader one (bag-of-words order/position-blindness
itself). A labeling fix that actually closes most of Part 1's gap would need to account for
n-gram/positional agreement with the ground truth, not just token-set or token-multiset overlap —
out of scope for this session, noted as the natural next step.

## Part 5: sequential position-aware labeling (user-proposed, full 3985-page A/B) — the real fix

Part 4's frequency-weighted variant only patched one narrow failure mode. The deeper problem, per
Parts 1-2, is that `overlap_labels()` has no concept of word order or position at all — and blocks
are confirmed deterministic and in document order (see "Determinism" above), so if ground-truth text
for genuine content *also* follows that order, keep/drop decisions can be made by **sequential
positional alignment** instead of bag-of-words overlap: search for each block's match starting from
where the previous accepted block's match ended, not anywhere in the whole document.

**Validated before implementing anything.** A stratified sample (3-5 pages, mixed high/low current
oracle score) from all 8 WCEB sub-datasets — not just the dragnet pages the idea was first
spot-checked on — showed a healthy monotonic-position majority everywhere (66-86% of consecutive
kept-block position pairs non-decreasing, no dataset breaking down, `cleaneval` — flagged upfront as
the riskiest, noisiest corpus — was mid-pack at 79.5%). This is measured on the *old* heuristic's
already-imperfect kept set (which includes false positives that break the sequence), so it's a
lower-bound signal, not a ceiling on how well a clean sequential method could do.

**Algorithm**: a forward-only cursor into the ground-truth token stream. Per block: search a bounded
window ahead of the cursor for the block's longest common subsequence (LCS) alignment (not a greedy
two-pointer match — provably worse on repeated tokens, e.g. block `[a,b,a]` vs. window `[b,a]`:
greedy finds length 1, true LCS finds 2). If match coverage clears the threshold, accept and advance
the cursor past the match; if not, reject and leave the cursor untouched so later blocks retry from
the same point. Short blocks (< `min_words`) get a smaller window and a near-contiguity requirement
instead of the general floor, to avoid short-phrase coincidences (e.g. a 3-word nav fragment
"Best of Toronto" spuriously matching a 2500-token article that happens to use "Toronto" once). A
"loose match" guard rejects any accepted-looking match whose span is disproportionate to the block's
own length (>3x) — a genuine positional match should be reasonably tight, not just technically
present somewhere in a wide window.

**A real bug was found and fixed during implementation, not just anticipated.** The new
`_lcs_with_endpoint()` primitive (`textMetrics.py`) needed both LCS length and *where* the match
ends, which the standard LCS recurrence's "always take the diagonal on a token match" shortcut
doesn't track correctly: when a block's phrase appears **twice** in the ground truth (a real case —
"Alison Santighian said:" launches two separate comments in one test page), the naive DP silently
preferred the *later* occurrence over the earlier one, over-advancing the cursor and stranding a
genuine 58-token comment body behind it — a measured regression on an already-good page (0.99 → 0.82
rouge5) before the fix. Root cause: the diagonal (match) branch overwrote the cell unconditionally
instead of comparing against the already-propagated equal-length alignment from the earlier
occurrence. Fixed by comparing every candidate transition (including the match case) and tie-breaking
toward the smaller endpoint, verified with a 1000-trial randomized cross-check against the existing
`_lcs_length()` (100% length agreement) plus a dedicated repeated-occurrence regression test. A first
attempt at a "retry with the full remaining window on a near-miss" refinement was tried and dropped
for the same class of reason: it let a low-quality match (an author byline not really present in the
article body) accept via a spurious, widely-scattered span far ahead in the token stream — the
loose-match span guard now rejects that outright rather than just discounting how far to advance.

**Worst-tail recovery** (same 41-page sample as Part 1, joined against Part 1's true-ceiling scores):
mean rouge5 **0.317 → 0.549** — a solid net gain, but not uniform. Big, clean wins dominate: several
dragnet pages hit 0.95-1.00 (up from 0.21-0.47), matching or nearly matching Part 1's true ceiling
for those exact pages (e.g. `dragnet/3e502bbb`: oracle 0.2087 → sequential 0.9884, identical to its
measured true ceiling). But some pages regress, concentrated on link-heavy, low-text pages where
ground truth itself is very short — one philosophy-blog directory page has only 65 ground-truth
tokens against 109 mostly-navigational blocks, too little signal for positional matching to have an
edge over the old heuristic's lucky guess. This mixed worst-tail picture is exactly why the rollout
decision below is per-dataset, not a blanket swap.

**Qualitative confirmation**: re-running the diff dump (`diagnoseOraclePages.py --dump-diff`) on the
worst pollution example from Part 2 confirms the fix works as intended, not just that a number moved
— `dragnet/3e502bbb`'s reconstruction-vs-ground-truth diff went from **+2888/-2** lines (the whole
nav menu and comment section) to **+1/-2** (essentially an exact match).

**Full 3985-page result** (`results/oracle/wceb_sequential.json`, compared via `compareArchs.py`):

| dataset | vs. old oracle | vs. weighted oracle | verdict |
|---|---|---|---|
| google-trends-2017 | +0.0789 | +0.0724 | significant, large |
| cleanportaleval | +0.0605 | +0.0563 | significant, large |
| dragnet | +0.0590 | +0.0538 | significant, large |
| scrapinghub | +0.0494 | +0.0463 | significant, large |
| l3s-gn1 | +0.0229 | +0.0204 | significant |
| readability | +0.0214 | +0.0189 | positive, not significant |
| cetd | +0.0085 | +0.0055 | positive, not significant |
| cleaneval | **−0.0374** | **−0.0384** | **significant regression** |
| **overall** | **+0.0260** | **+0.0226** | **significant** (95% CI [+0.0211, +0.0309] vs. old) |

Overall ROUGE-5: **0.9013 (old) → 0.9047 (weighted) → 0.9273 (sequential)**. Sequential's overall
gain (+0.026) is **~7.5x** the weighted variant's (+0.0035) — this is the dominant fix found in this
investigation, not a marginal refinement. It significantly improves 5 of 8 datasets, is
directionally positive but not significant on 2 more, and significantly *regresses* cleaneval alone.

**Rollout decision, per the "no in-function fallback" design principle**: don't patch cleaneval's
regression into the labeling function itself (a fallback there would reintroduce exactly the
diagnosed problem for the datasets sequential is genuinely better on). Adopt `overlap_sequential` for
new work; treat which labeler an oracle/analysis run should use as a **per-dataset choice**, made
explicitly at the `--labeler`/`--datasets` CLI level (`evaluateOracleCeiling.py`), not silently
resolved inside the function.

**Explicit scope limits** (unchanged from the design, restated for the record): this only changes
the oracle diagnostic. It does **not** improve the real training-label pipeline
(`cacheEmbeddingsForCombined.py`/`cacheEmbeddings.py` have their own separate, unordered-set-based
labeling logic, not validated for the order premise) or the block-level P/R/F1 reference every
trained model is scored against (`evalCommon.py`, still on the plain heuristic) — both are natural,
separately-scoped follow-ups. Segmentation-miss pages (Part 2's `l3s-gn1/b78d17d3` example, and this
part's philosophy-blog directory page) remain a distinct, unfixable-by-labeling-changes category.

## Summary

1. **Determinism**: checked, ruled out — block segmentation is fully deterministic.
2. **Dominant driver**: the order/frequency-blind bag-of-words silver-label heuristic makes
   genuinely bad block-selection decisions on hard pages — worst-tail true-ceiling headroom
   averages **+0.39 ROUGE-5**, reaching +0.58 on dragnet specifically, with individual pages
   recovering from ~0.2–0.5 up to 0.95–1.0 at the *same* block granularity. This is a heuristic
   problem, not a segmentation or reconstruction problem.
3. **Secondary, confirmed driver**: bare table-cell/list-item reconstruction losing structural
   context, real and statistically significant in the three hardest datasets, but an order of
   magnitude smaller effect (−0.01 to −0.05) than the labeling gap above.
4. **Confirmed, partial fix**: a frequency-weighted (clipped-multiset) labeling variant gives a
   real, statistically significant improvement (0.9013 → 0.9047, +0.0035, 95% CI [+0.0028,
   +0.0041], significant on 7/8 datasets) — correctly strongest on dragnet and google-trends-2017,
   the exact datasets Parts 1/2 flagged. But it recovers only ~1% of Part 1's +0.39 worst-tail
   headroom: it fixes repetition-driven false positives, not the deeper order/position-blindness of
   bag-of-words overlap itself.
5. **Confirmed, dominant fix (Part 5)**: sequential position-aware labeling — using the fact that
   blocks and genuine ground-truth content share document order — gives **0.9013 → 0.9273**
   overall (+0.0260, 95% CI [+0.0211, +0.0309], significant), **~7.5x** the weighted variant's
   effect. Significantly improves 5/8 datasets (largest: google-trends-2017 +0.079, cleanportaleval
   +0.061, dragnet +0.059, scrapinghub +0.049), positive but not significant on 2 more, and
   significantly *regresses* cleaneval alone (−0.037) — not patched with an in-function fallback
   (would reintroduce the diagnosed problem elsewhere), instead treated as a per-dataset rollout
   decision via the `--labeler`/`--datasets` CLI flags. On the worst tail specifically: mean rouge5
   0.317 → 0.549, with several pages reaching their Part-1-measured true ceiling almost exactly
   (`dragnet/3e502bbb`: 0.21 → 0.99, diff shrank from +2888/-2 lines to +1/-2). Only affects the
   oracle diagnostic — does not (yet) touch the actual WebMainBench training-label pipeline or
   `evalCommon.py`'s P/R/F1 reference, both natural follow-ups.
6. **Ruled out**: the ROUGE-N zero-token cliff (3/3985 pages), corpus-wide.
7. **Practical implication**: the ~0.90 oracle ceiling was never a hard pipeline limit — it was
   capped by the labeling heuristic's fidelity to the actual ROUGE-5 objective, and most of the
   demonstrated headroom (Part 1's +0.39) turns out to be closable with a heuristic that's still
   just a heuristic (no ground truth needed at decision time, unlike Part 1's diagnostic search) —
   sequential position-aware labeling recovers the large majority of it on 5-7 of 8 datasets.
   cleaneval is the clear exception and warrants its own investigation before adopting sequential
   there. This reframes the earlier `CLAUDE.md`/`overview.md` framing further: not just "GRU/xLSTM
   already meet the old, flawed oracle" but "the oracle itself had much more room to improve than
   0.90 suggested, and a large fraction of that room was recoverable without any ground-truth
   peeking, via a smarter but still fully causal labeling rule." Whether porting this idea to actual
   training-label generation (currently untouched) would improve real model quality, not just the
   diagnostic ceiling, is the natural next question.

## Part 6: wiring the fix into actual training data (Phase A + Phase B)

Part 5's fix only touched the oracle diagnostic. Phase B's real LODO training data
(`cacheEmbeddingsFromWCEB.py`) turned out to import `blockReconstruction.overlap_labels` directly —
literally the same function — so the fix applies there unmodified. Phase A's real training data
(`cacheEmbeddingsForCombined.py`, WebMainBench) has a separate, independently-duplicated word-overlap
heuristic against `cc-select="true"`-marked ground truth pooled into a word bag.

**Found**: `mineru_html.process.simplify_html()` (the vendored dependency) already propagates
`cc-select` onto whichever mapping-tree node ends up carrying `_item_id` — confirmed by reading
`process_paragraphs()` directly and by testing on real WebMainBench rows. This means a training
block's content/boilerplate label can be read directly off the mapping tree's attribute, no
word-overlap heuristic needed, for any page with `cc-select` markup at all.

**Measured** (150 sampled WebMainBench pages, 17,541 blocks): the old word-overlap method marks
49.7% of all blocks as content; direct DOM correspondence marks 29.9%. 23.3% of all blocks disagree,
~12x more often in the direction of the old method over-including boilerplate (nav breadcrumbs, a
bare "Home" link, a sports-scores widget — the same over-inclusion character found throughout this
investigation). Separately (300 pages, 9,747 marked blocks): the propagation mechanism itself is
reliable — only ~0.6% of marked blocks have a small marked fragment dragging a much bigger unrelated
container along (a known, accepted, unfixed residual — not worth the added complexity).

**Built**: `dom_correspondence_labels()` in `cacheEmbeddingsForCombined.py` (`--labeler dom`), reused
by `blockDataset.py`/`trainTrainableMiniLM.py` (`--labeler dom`) so TrainableMiniLM's training data
generation matches. Falls back to the pre-existing `convert_main_content`/`groundtruth_content`/
`main_html` word-overlap path only for pages with zero `cc-select` markup at all. Every `.npz`
records which labeler produced it (`label_method`), and unmatched blocks (no mapping-tree
counterpart — a real, `process_paragraphs`-internal possibility, not just theoretical) are tracked
and warned on, not silently defaulted. Validated locally on the same 300-page sample: **0%
unmatched**, 29.2% marked (matches the 150-page sample's 29.9% closely).

### Runbook: rebuilding the training caches (Gruenau)

Both phases go to **new** `--out` directories — never mix old- and new-labeled `.npz` files in one
directory, since `trainCommon.list_cache_files()` globs an entire directory with no concept of which
labeler produced which file.

**Phase A** (WebMainBench, full rebuild):
```
python -m src.caching.cacheEmbeddingsForCombined \
    --jsonl src/caching/webmainbench.jsonl --labeler dom --out cache_dom/
```

**Phase B** (WCEB/LODO, two invocations — `cleaneval` significantly regresses under
`overlap_sequential`, kept on the old labeler; both write into the same new directory, filenames
disambiguate by dataset):
```
python -m src.caching.cacheEmbeddingsFromWCEB \
    --wceb src/evaluation/web-content-extraction-benchmark/datasets/combined \
    --labeler overlap_sequential \
    --datasets cetd cleanportaleval dragnet google-trends-2017 l3s-gn1 readability scrapinghub \
    --out cache_dom/
python -m src.caching.cacheEmbeddingsFromWCEB \
    --wceb src/evaluation/web-content-extraction-benchmark/datasets/combined \
    --labeler overlap --datasets cleaneval --out cache_dom/
```

**TrainableMiniLM** (no `.npz` cache — reads WebMainBench directly at training start):
```
python -m src.models.trainableminilm.trainTrainableMiniLM \
    --jsonl src/caching/webmainbench.jsonl --labeler dom --epochs 15
```

**Landmine, not obvious from the error output**: both caching scripts skip any page whose output
file already exists (`if os.path.exists(path): continue`). If a labeler choice needs to change after
the fact — e.g. re-running Phase B's `cleaneval` invocation with a different flag — **delete the
stale `wceb-<dataset>-*.npz` files first**. Re-running with a different `--labeler` against an
already-populated directory silently no-ops, leaving old-labeled files in place with no error or
warning.

BiLSTM/GRU/xLSTM's `trainLSTM.py`/`trainGRU.py`/`trainxLSTM.py` need no code changes — point
`--cache` at the new `cache_dom/` directory instead of the old `cache/`.

**Not done here** (deliberately out of scope): `evalCommon.py`'s block-level P/R/F1 reference for
trained models still uses the old `overlap_labels` — changing it would make every previously-reported
P/R/F1 number non-comparable to future runs, a separate decision. Actually running the retraining and
the follow-up `analysis/collect_data.py`/`generate_figures.py` comparison pass is the natural next
session, once results come back from the cluster.
