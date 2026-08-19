# BiLSTM

> **Labeling note (2026-08-19)**: Phase A training data here used the original
> word-overlap silver-labeling heuristic (WebMainBench `cc-select` word-bag overlap,
> `cacheEmbeddingsForCombined.py`). Phase B/LODO used WCEB's `overlap_labels`. Both were
> found to have real, measured flaws — see `analysis/oracle_investigation.md`. A
> DOM-correspondence relabeling (Phase A, `--labeler dom`) and a sequential
> position-aware relabeling (Phase B, `--labeler overlap_sequential`) are built and
> ready to rerun; this document has not yet been updated with results from either.

Source data: `results_from_server/runs/sweep_bilstm/`, `results_from_server/runs/lodo_bilstm/`,
`results_from_server/results_combined{3,4}/`, `results_from_server/results/`. Field definitions:
`docs/RESULTS.md`. Every number below traces to `analysis/consolidated_data.json` (built by
`analysis/collect_data.py`), which in turn is read straight from those files.

## Architecture

Plain `nn.LSTM(input_dim=431, hidden_dim, num_layers=1, bidirectional=True)` +
`Linear(hidden_dim*2, 1)` classifier head over the 431-dim combined block embedding, one block
per timestep. At the sweep-winning config (`hidden_dim=128`) that's **574,721 parameters** — the
baseline of the three architectures, and the one with the longest history in this repo (the
"combined3/4" legacy runs below predate the sweep infrastructure by six weeks).

## Sweep: hyperparameter search

Grid: `hidden_dim {64,128,256} × dropout {0.1,0.3,0.5} × lr {0.001,0.0005}` = 18 configs, ranked
by held-out validation block-F1 (`val_f1`), top-5 given a full 3985-page WCEB pass.

**Winner**: `hidden_dim=128, dropout=0.3, lr=0.0005` — `val_f1=0.9000`, `wceb_rouge5=0.7183`,
training wall-clock 425.8s.

**The cheap proxy metric (`val_f1`) does not reliably predict the expensive one
(`wceb_rouge5`) for this architecture.** The five top-K configs' `val_f1` values are clustered in
a 0.0023-wide band (0.8999–0.9022) — essentially a statistical tie — while their `wceb_rouge5`
scores span 0.111 (0.607–0.718), an 11-point range:

| rank by val_f1 | hidden/dropout/lr | val_f1 | wceb_rouge5 | rank by wceb_rouge5 |
|---|---|---|---|---|
| 1 | 256 / 0.5 / 0.001 | 0.9022 | 0.6072 | 5 (worst) |
| 2 | 128 / 0.3 / 0.001 | 0.9009 | 0.6344 | 2 |
| 3 | 256 / 0.1 / 0.001 | 0.9001 | 0.6334 | 3 |
| 4 | 128 / 0.3 / 0.0005 (**winner**) | 0.9000 | **0.7183** | **1** |
| 5 | 64 / 0.5 / 0.001 | 0.8999 | 0.6276 | 4 |

The config with the *best* val_f1 has the *worst* full-benchmark ROUGE-5; the eventual winner
ranked 4th of 5 on the cheap metric. This isn't noise in the sweep, it's a property of what
`val_f1` measures: it's block-level precision/recall on a held-out split at a fixed 0.5 threshold,
and it saturates once a config is "good enough" at the coarse keep/drop decision. ROUGE-5 is far
more sensitive to exactly where the model draws that line on borderline blocks, because it needs
5 contiguous matching tokens — a single wrongly-dropped or wrongly-kept block at a page boundary
can break several 5-grams at once (this echoes the boundary-noise sensitivity already documented
in the ROUGE-5-vs-ROUGE-L investigation notes in `CLAUDE.md`). **Practical implication: for
BiLSTM, `val_f1` alone is not a trustworthy ranking signal — the sweep's two-stage design (rank
cheaply, verify expensively on top-K) is doing real work here, not just saving compute.**
See `figures/sweep_sensitivity.png`.

## Phase A: WebMainBench → full WCEB (in-domain-trained, cross-domain-evaluated)

Overall ROUGE-5 F1 = **0.7183**, ROUGE-L F1 = **0.7422**, block-level P/R/F1 = **0.927 / 0.419 /
0.577**, throughput 4.28 pages/sec (0.234s/page mean).

Per-dataset ROUGE-5: cetd 0.780, cleaneval 0.701, cleanportaleval 0.753, dragnet 0.698,
google-trends-2017 0.560 (hardest), l3s-gn1 0.789 (easiest), readability 0.556, scrapinghub 0.704.

**The block-level numbers explain the gap better than the ROUGE numbers alone**: precision is
high (0.927) but recall is low (0.419) — BiLSTM is *conservative*, correctly keeping most of what
it does keep, but missing well over half the true content blocks (98,262 TP vs 136,474 FN). This
recall problem is not specific to the winning config — every top-K bilstm config in the sweep has
block-F1 in the 0.48–0.58 band (see leaderboard above), so it's a property of the architecture at
this training setup, not a one-off underfit run. Compare this to GRU's block-level R=0.884 at a
similar precision (0.935, see `gru.md`) — same 431-dim input, same loss (`BCEWithLogitsLoss` with
the same `pos_weight` class-imbalance handling), same 0.5 keep threshold, radically different
recall. Given identical training mechanics, the difference is architectural: LSTM's extra gate and
separate cell state appear to push the decision boundary toward "when in doubt, drop" more than
GRU's simpler update does. This is an empirical pattern in the data, not a confirmed mechanism —
worth flagging as an open question rather than asserting the causal story with more confidence
than the evidence supports.

## Comparison to legacy pre-sweep baselines

Three earlier evaluations of the same architecture (all pre-date `sweep.py`/LODO, June 2026,
manually-set hyperparameters, `model_combined.pt`, same 574,721 params):

| run | date | overall ROUGE-5 | block F1 | pages/sec |
|---|---|---|---|---|
| `results_combined3/wceb.json` | 2026-06-08 01:01 | 0.6453 | 0.585 | 4.55 |
| `results_combined4/wceb.json` | 2026-06-08 01:23 | 0.6557 | 0.684 | 4.72 |
| `results/wceb_minilm.json` (plain MiniLM, non-combined) | 2026-06-06 | 0.6333 | — | — |
| **sweep-tuned Phase A (this run)** | 2026-07-24 | **0.7183** | 0.577 | 4.28 |

The systematic hyperparameter sweep (hidden_dim/dropout/lr) plus early stopping on val-F1
(`--patience`, not used in the ad-hoc June runs) buys **+6.3 to +8.5 ROUGE-5 points** over the
manually-configured legacy runs — a real, attributable improvement from tuning, not just
re-running the same config. Note block-F1 doesn't move the same direction (combined4's 0.684 >
this run's 0.577): the legacy runs were evaluated at a different implicit operating point (no
`keep_threshold` field recorded, block-level metric logic itself changed on 2026-07-24 per the
`textMetrics.py`/`blockReconstruction.py` bugfix — see `CLAUDE.md`), so block-F1 isn't safely
comparable across this boundary; ROUGE-5/ROUGE-L are the metrics to trust for this before/after
comparison. See `figures/legacy_vs_current_bilstm.png`.

## Phase B: LODO (domain-adapted, held out one WCEB dataset at a time)

Same winning config, retrained 8 times excluding one WCEB sub-dataset from training each time,
evaluated only on the held-out fold (`runs/lodo_bilstm/aggregate.json`):

| fold | rouge5 | rouge_l | block_f1 | pages/sec |
|---|---|---|---|---|
| cetd | 0.9306 | 0.9432 | 0.8230 | 4.62 |
| cleaneval | 0.8402 | 0.8726 | 0.8651 | 6.00 |
| cleanportaleval | 0.8587 | 0.8710 | 0.7016 | 5.58 |
| dragnet | 0.8288 | 0.8527 | 0.8131 | 4.82 |
| google-trends-2017 | 0.7190 | 0.7479 | 0.5673 | 2.58 |
| l3s-gn1 | 0.8706 | 0.8734 | 0.7462 | 7.25 |
| readability | 0.8488 | 0.8612 | 0.8186 | 4.64 |
| scrapinghub | 0.8589 | 0.8700 | 0.7786 | 6.16 |
| **mean ± std** | **0.8445 ± 0.0592** | 0.8615 ± 0.0536 | 0.7642 ± 0.0943 | — |

**Phase B (0.8445) is +0.126 ROUGE-5 over Phase A (0.7183) — a much bigger jump than GRU
(0.907→0.847, a *drop* of 0.060) or xLSTM (0.902→0.833, a drop of 0.069) see under the same
LODO procedure.** The direction of this effect is architecture-dependent, and it makes sense given
the recall problem documented above: BiLSTM's WebMainBench-only Phase-A recall was low because it
hadn't seen enough WCEB-style content-block patterns; adding 7/8 WCEB datasets to training data
directly targets that gap, and BiLSTM has by far the most headroom to close. GRU/xLSTM were
already recall-strong from WebMainBench alone (see their docs), so the same LODO retraining has
comparatively little upside for them and mostly reflects the cost of training on a smaller,
narrower dataset. `google-trends-2017` is the hardest fold for every architecture tried (smallest
WCEB sub-dataset, likely the most distributionally different) — see `figures/lodo_per_fold.png`.

## Oracle headroom

Oracle ceiling (full 3985-page, gold token-overlap labels, no model): overall ROUGE-5 = 0.9013.
Recomputed locally (`analysis/stats/oracle_vs_bilstm.md`, full WCEB, not the partial 263-page
version): **mean diff (oracle − bilstm) = +0.1830, 95% CI [+0.1738, +0.1925], significant on every
one of the 8 WCEB sub-datasets** — BiLSTM Phase A leaves substantial, statistically confirmed
headroom on the table (worst gap: readability +0.360, cetd smallest gap +0.139). Phase B narrows
this (0.8445 vs 0.9013 ≈ 0.057 gap) but the recomputed oracle comparison above is Phase-A-vs-oracle
only — a Phase-B-vs-oracle comparison would need each fold's own held-out oracle slice, not done
here.

## Caveats

- `val_f1` is not a reliable ranking signal for this architecture (see sweep section) — any future
  BiLSTM sweep should either widen top-K or add a cheap WCEB proxy metric.
- Legacy-run block-F1 numbers are not directly comparable to current ones (metric logic changed
  2026-07-24); only trust ROUGE-5/ROUGE-L for the legacy-vs-current comparison.
- The precision/recall architecture explanation above is a pattern in the data (consistent across
  every sweep config and both phases), not something independently verified by ablation — flagged
  as an open question for further investigation, not a settled conclusion.
