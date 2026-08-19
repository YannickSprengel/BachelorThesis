# Cross-architecture overview

> **Labeling note (2026-08-19)**: every number below (all four architectures) was
> produced with the original word-overlap silver-labeling heuristics, both found to
> have real, measured flaws — see `analysis/oracle_investigation.md` and each
> architecture doc's own labeling note. Fixed labelers are built and ready to rerun on
> the cluster; nothing here reflects them yet.

Synthesis of `bilstm.md`, `gru.md`, `xlstm.md`, `trainable_minilm.md`. Read those first for the
per-architecture "why"; this file is the cross-cutting story for the thesis. Source data:
`results_from_server/results/summary.{json,md}`, `compare_*.json`, `oracle.json`, the locally
recomputed `analysis/stats/oracle_vs_*.{json,md}` and `trainable_minilm_vs_bilstm.{json,md}`, and
`results_from_server/trainable_minilm/wceb.json`. Field definitions: `docs/RESULTS.md`.

## Headline numbers

| arch | params | Phase A rouge5 | Phase A rouge_l | pages/sec | sweep train wall (winner) | Phase B rouge5 (mean±std) |
|---|---|---|---|---|---|---|
| bilstm | 574,721 | 0.7183 | 0.7422 | 4.28 | 426s | 0.8445 ± 0.0592 |
| gru | 431,105 | **0.9068** | **0.9226** | **5.23** | 894s | 0.8470 ± 0.0506 |
| xlstm | 671,393 | 0.9025 | 0.9174 | 4.91 | **4046s** | 0.8334 ± 0.0666 |
| trainable-minilm-bilstm | 118,228,481 | 0.8675 | 0.8865 | 2.891 | 57,358s | n/a¹ |
| oracle | — | 0.9013 | 0.9148 | — | — | — |

¹ No Phase B for trainable-minilm-bilstm: not a "not yet run" gap like GRU's temporary one was,
but a structural absence — there is no WCEB-training-data ingestion path for this architecture at
all, so LODO cannot be run against it. See `trainable_minilm.md`.

## Finding 1: architecture matters a lot pre-domain-adaptation, and mostly washes out after it

This is the central result of the whole comparison, not "GRU wins." On Phase A (trained on
WebMainBench only, evaluated cross-domain on WCEB), architecture choice is worth up to **+0.19
ROUGE-5** (GRU/xLSTM vs BiLSTM) — confirmed significant on the full 3985-page benchmark:

- bilstm vs gru: mean diff −0.1885, 95% CI [−0.1975, −0.1797]
- bilstm vs xlstm: mean diff −0.1842, 95% CI [−0.1933, −0.1755]
- gru vs xlstm: mean diff +0.0043, 95% CI [+0.0018, +0.0068] — significant but tiny; GRU edges
  out xLSTM, but not on every individual WCEB sub-dataset (see `results/compare_gru_vs_xlstm.md`)

Once WCEB-domain data enters training (Phase B / LODO), all three converge to a much narrower
band: 0.8334–0.8470. BiLSTM *gains* +0.126 ROUGE-5 from this; xLSTM *loses* 0.069; GRU *loses*
0.060 — a similar direction and magnitude to xLSTM's, now confirmed with verified per-fold data
(`gru.md`), not just a provisional estimate. The mechanism proposed in the per-architecture docs: BiLSTM's Phase-A block-level
recall (0.419) is far below GRU/xLSTM's (0.884/0.873) under identical training mechanics (same
431-dim input, same `pos_weight`-weighted BCE loss, same 0.5 threshold) — BiLSTM has the most
recall headroom to close with in-domain data, and closes most of it; GRU/xLSTM were already
recall-strong, so the same procedure mostly just trades a larger, more diverse training set
(WebMainBench) for a smaller, narrower one (WCEB minus one fold), a net cost for them. **Practical
framing for the thesis: if in-domain (WCEB-like) training data is available, architecture choice
matters much less than whether that data was used at all — but if it's not available, GRU/xLSTM's
architectural properties are doing real, load-bearing work that BiLSTM's is not.**

## Finding 2: quality-vs-cost trade-off is not a straight line — GRU dominates both axes

The thesis frames this as a trade-off, and it usually is one (spend more, get more) — but on this
data, GRU is simultaneously the smallest (431K params, 25% smaller than BiLSTM), fastest
(5.23 pages/sec inference, fastest to train per epoch among the tuned winners at 894s), *and*
highest-quality (0.9068 Phase-A ROUGE-5) of the three. It is not on the Pareto frontier trading
against BiLSTM or xLSTM — it strictly dominates both on this benchmark. xLSTM is the clearest
counter-example of a real trade-off paid without full return: essentially tied with GRU on quality
(0.9025 vs 0.9068, a 0.0043-point gap that's itself only marginally significant) but with more
parameters and **~4.5x GRU's / ~9.5x BiLSTM's training wall-clock** (4046s vs 894s vs 426s for
the respective winning configs; xLSTM's *cheapest* swept config, 2012s, still exceeds every
BiLSTM config and all but two GRU configs in the whole grid). BiLSTM is the cheapest to train but
pays for it in Phase-A quality — though that gap is largely recoverable with in-domain data
(Finding 1). See `figures/quality_vs_cost.png` and `figures/training_cost.png`.

Methodological caveat on all wall-clock/throughput numbers in this analysis: training was run
directly in cluster tmux sessions, not through a resource-managed batch allocation, so there's no
record of whether the GPU was exclusively held during any given run (see `gru.md`'s LODO
throughput caveat for a concrete instance). The gaps reported above are large enough (2-10x) that
they're very unlikely to be contention artifacts, but treat these as relative-cost indicators
rather than controlled benchmarks, particularly for any closer comparison than this one.

## Finding 3: the oracle ceiling is close to fully closed by GRU and xLSTM on Phase A

Recomputed locally on the full 3985-page WCEB set (`analysis/stats/oracle_vs_*.md`), replacing
the existing partial 263-page `compare_oracle_vs_xlstm.md`:

| comparison | mean diff (oracle − model) | 95% CI | verdict |
|---|---|---|---|
| oracle vs bilstm | +0.1830 | [+0.1738, +0.1925] | oracle significantly ahead |
| oracle vs gru | **−0.0055** | [−0.0083, −0.0026] | **model significantly ahead of oracle** |
| oracle vs xlstm | −0.0012 | [−0.0042, +0.0018] | not significant (statistical tie) |

GRU and xLSTM's Phase-A results are, in aggregate, at or above the oracle ceiling — meaning the
"headroom to a perfect tagger" framing from the earlier `CLAUDE.md` investigation (oracle ~0.90,
BiLSTM ~0.75, "~17-point gap... real headroom for a better-tuned tagger") has been **fully
resolved by architecture choice + tuning**, not just narrowed. This also surfaces something more
interesting than "the models are great": since "oracle" here means gold *silver-label* keep/drop
decisions run through the same reconstruction pipeline (not a true human ROUGE-5-optimal
selection), a well-tuned tagger scoring *above* it on some datasets (GRU: cetd, dragnet; xLSTM:
cetd, dragnet — both significantly, while oracle significantly beats both on cleaneval and
google-trends-2017 instead) shows the silver-label heuristic is an imperfect
proxy for "which blocks maximize ROUGE-5," not that either model has solved the problem. **The
remaining ~1-point differences between GRU/xLSTM and oracle are now within the noise floor of the
labeling heuristic itself, not a model-capacity signal** — further gains on this benchmark likely
require improving the ground-truth/labeling methodology, not training a bigger tagger. Only
BiLSTM still has genuine, statistically unambiguous headroom left (+0.183).

**`analysis/oracle_investigation.md` confirms this with hard evidence, not just plausibility —
and revises the "GRU/xLSTM already beat the oracle" framing above.** Block segmentation is
deterministic (checked and ruled out as a cause). The oracle's own silver-label heuristic
(order/frequency-blind bag-of-words overlap) was demonstrably far from ROUGE-5-optimal: a greedy
re-search at the *same* block granularity found **+0.39 mean ROUGE-5 headroom** on the oracle's
worst-scoring pages (+0.58 on dragnet). A cheap frequency-weighted fix recovered only a sliver of
that (+0.0035 overall). A **sequential position-aware labeling fix** — using the fact that blocks
and genuine ground-truth content share document order — recovers far more: **overall oracle
ROUGE-5 0.9013 → 0.9273** (+0.0260, significant, ~7.5x the weighted fix's effect), significant on
5/8 datasets, with one regression (cleaneval, −0.037, not yet resolved). **This puts the properly-
labeled oracle (0.9273) clearly back above both GRU (0.9068) and xLSTM (0.9025)** — the "models at
or above oracle" result above was an artifact of the old oracle's flawed labeling, not evidence the
models had closed all real headroom. The noise-floor/labeling-methodology conclusion above still
holds directionally (further gains need better labels, not bigger taggers), but the size of the
remaining gap was understated by the old oracle number.

## Finding 4: fine-tuning the embedder closes most of the BiLSTM→GRU/xLSTM gap, at a steep price

`trainable_minilm.md` covers this in full; summary here. Swapping BiLSTM's frozen 431-dim combined
embedding for an end-to-end fine-tuned MiniLM encoder (same tagger, same loss, same threshold)
lifts Phase-A ROUGE-5 from 0.7183 to **0.8675** — confirmed significant against frozen
combined-BiLSTM (`analysis/stats/trainable_minilm_vs_bilstm.json`: mean diff +0.1492, 95% CI
[+0.1405, +0.1581], significant on all 8 WCEB sub-datasets) and against the frozen 384-dim plain-
MiniLM legacy baseline (0.6333 → 0.8675, +0.234, same tagger and base encoder, isolating the
fine-tuning variable alone). The mechanism matches Finding 1's recall story: BiLSTM's frozen-
embedding Phase-A recall (0.419) was the bottleneck, and fine-tuning lifts recall to 0.7114 —
short of GRU/xLSTM's 0.884/0.873, but most of the way there.

It does not fully close the gap to GRU/xLSTM (0.8675 vs 0.9068/0.9025) or to the oracle ceiling
(mean diff oracle − model = +0.0338, 95% CI [+0.0291, +0.0385], significant —
`analysis/stats/oracle_vs_trainable_minilm.json`), and the cost is disproportionate to the gain:
~274x GRU's parameter count and ~64x GRU's training wall-clock (57,358s vs. 894s) for a result
still below GRU's. **On this benchmark, fine-tuning the embedder is not a Pareto improvement over
picking a better frozen-embedding architecture** — GRU/xLSTM reach equal-or-better quality far more
cheaply (Finding 2). The one genuine advantage this run has over every other architecture's Phase A
number: its WCEB evaluation is *structurally* leak-free (no WCEB-training-data path exists for this
architecture at all), stronger than the "trained on whatever's in `cache/`" guarantee the other
three Phase A numbers carry. Caveat: this is one manually-chosen config with no sweep, so "does
fine-tuning help" and "did we pick good hyperparameters" aren't cleanly separated the way BiLSTM's
sweep separated them. See `figures/frozen_vs_finetuned_minilm.png` and `figures/finetuning_cost.png`.

## Status: architectures without usable results

- **Transformer** (`src/models/transformer/`): training/eval code exists
  (`trainTransformer.py`, `evaluateTransformer.py`) and it appears in `smokeTest.py`'s 2-epoch
  integration check (val_f1=0.2857, worst of the four smoke-tested architectures), but no
  sweep or LODO results exist in this download — not yet run at scale.

## Data caveats index

- ~~GRU Phase B (LODO) — unverified.~~ **Resolved 2026-08-18**: `runs/lodo_gru/` reran on the
  cluster and re-synced; verified per-fold data now backs the 0.8470±0.0506 number above (see
  `gru.md`). The earlier `summary.json`-only estimate (0.8474±0.0569) was a close but stale
  snapshot, superseded by the fresh `aggregate.json`.
- **xLSTM sweep ran twice** (36 dirs); only the second, complete batch (2026-07-28/29) has
  `wceb.*` and feeds the numbers above. The first batch is an aborted attempt, not a second grid.
- **Legacy xLSTM baseline used `keep_threshold=0.3`**, not the current default 0.5 — the
  legacy-vs-current improvement in `xlstm.md` conflates architecture/tuning gains with a threshold
  change.
- **Legacy BiLSTM block-F1 numbers are not comparable to current ones** — the block-level metric
  logic changed on 2026-07-24 (`textMetrics.py`/`blockReconstruction.py` bugfix). Only trust
  ROUGE-5/ROUGE-L for the legacy-vs-current BiLSTM comparison.
- **`results_from_server/results/oracle/wceb.csv` is a partial download** (263 rows, cetd only)
  despite `oracle.json` reflecting the full 3985-page run — the complete CSV exists locally at
  repo-root `results/oracle/wceb.csv` and was used for every oracle recomputation in this
  analysis.
- No **ablation-scale** (`ablationScale.py`, scaled-up-BiLSTM-vs-baseline) results were present in
  this download — not covered anywhere in this analysis.
