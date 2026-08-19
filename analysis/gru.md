# GRU

> **Labeling note (2026-08-19)**: Phase A training data here used the original
> word-overlap silver-labeling heuristic (WebMainBench `cc-select` word-bag overlap,
> `cacheEmbeddingsForCombined.py`). Phase B/LODO used WCEB's `overlap_labels`. Both were
> found to have real, measured flaws — see `analysis/oracle_investigation.md`. A
> DOM-correspondence relabeling (Phase A, `--labeler dom`) and a sequential
> position-aware relabeling (Phase B, `--labeler overlap_sequential`) are built and
> ready to rerun; this document has not yet been updated with results from either.

> **Not documented in `CLAUDE.md`** — `src/models/gru/` is a third architecture added to the
> pipeline on 2026-07-28 (commit `b935270`), fully wired into `sweep.py`/`aggregateLODO.py`
> alongside BiLSTM/xLSTM, not a stub.

Source data: `results_from_server/runs/sweep_gru/`, `results_from_server/runs/lodo_gru/`,
`results_from_server/results/`. Field definitions: `docs/RESULTS.md`. Every number below traces
to `analysis/consolidated_data.json` (built by `analysis/collect_data.py`).

## Architecture

`nn.GRU(input_dim=431, hidden_dim, num_layers=1, bidirectional=True)` +
`Linear(hidden_dim*2, 1)` — structurally an almost line-for-line swap of BiLSTM's `nn.LSTM` for
`nn.GRU`, same interface, same `predict_document()` signature. The module's own docstring is
explicit this is deliberate: GRU has 3 gates vs. LSTM's 4 (no separate cell state), so at the same
`hidden_dim` it's naturally cheaper — **not capacity-matched to BiLSTM on purpose**, since the size
difference is itself a data point for the thesis's quality-vs-cost comparison. At the sweep-winning
config (`hidden_dim=128`) that's **431,105 parameters — 75% of BiLSTM's 574,721** at the same
`hidden_dim`.

## Sweep: hyperparameter search

Same grid shape as BiLSTM: `hidden_dim {64,128,256} × dropout {0.1,0.3,0.5} × lr {0.001,0.0005}`
= 18 configs.

**Winner**: `hidden_dim=128, dropout=0.1, lr=0.0005` — `val_f1=0.8857`, `wceb_rouge5=0.9068`,
training wall-clock 894.0s.

**Unlike BiLSTM, GRU is robust to hyperparameters on this task.** All four top-K configs land
within a 0.0043-wide `wceb_rouge5` band (0.9025–0.9068) regardless of `hidden_dim` (even
`hidden_dim=64`, less than a third of the winner's 431K params at 190,977 params, scores
0.9037 — within noise of the 128-dim winner):

| hidden/dropout/lr | val_f1 | wceb_rouge5 | block_f1 | params |
|---|---|---|---|---|
| 128 / 0.3 / 0.0005 | 0.8872 | 0.9025 | 0.9183 | 431,105 |
| 128 / 0.5 / 0.0005 | 0.8865 | 0.9038 | 0.9063 | 431,105 |
| 128 / 0.1 / 0.0005 (**winner**) | 0.8857 | 0.9068 | 0.9086 | 431,105 |
| 64 / 0.5 / 0.0005 | 0.8856 | 0.9037 | 0.9060 | 190,977 |

This is the opposite pattern from BiLSTM's sweep (see `bilstm.md`), where `val_f1` and
`wceb_rouge5` rankings actively disagreed. Here they're both just flat — GRU reaches a similar,
high-quality operating point across most of the grid, so hyperparameter choice mostly trades off
training wall-clock (152.7s–2123.5s across the full 18-config grid) rather than final quality.
See `figures/sweep_sensitivity.png`.

## Phase A: WebMainBench → full WCEB (in-domain-trained, cross-domain-evaluated)

Overall ROUGE-5 F1 = **0.9068**, ROUGE-L F1 = **0.9226**, block-level P/R/F1 = **0.935 / 0.884 /
0.909**, throughput **5.23 pages/sec** (0.191s/page mean) — this is simultaneously the best
quality, fewest parameters, and fastest inference of the three architectures on Phase A. Worth
stating plainly since it's the standout result of this whole comparison: on the
quality-vs-computational-cost trade-off the thesis is built around, GRU dominates on both axes
here, not just one.

Per-dataset ROUGE-5: cetd 0.933, cleaneval 0.904, cleanportaleval 0.886, dragnet 0.898,
google-trends-2017 0.815 (hardest, as for every architecture), l3s-gn1 0.925, readability 0.919,
scrapinghub 0.911.

Block-level recall (0.884) is dramatically higher than BiLSTM's (0.419) at similar precision
(0.935 vs 0.927) — same 431-dim input, same `BCEWithLogitsLoss`+`pos_weight` class-imbalance
handling, same 0.5 keep threshold as BiLSTM. This is the same pattern noted in `bilstm.md`: GRU's
simpler gating appears to reach a much less conservative, higher-recall operating point than LSTM
under otherwise-identical training mechanics. No legacy pre-sweep baseline exists for GRU (new
architecture as of 2026-07-28), so there's no "improvement over previous runs" comparison to make
here — Phase A numbers above are the first and only evaluation of this architecture.

## Phase B: LODO — domain-adapted, held out one WCEB dataset at a time

> **Verified 2026-08-18.** The initial download had `runs/lodo_gru/` with all 8 fold directories
> present but completely empty (no `config.json`/`metrics.json`/`model.pt`/`wceb.*`, no
> `aggregate.json`), so an earlier pass of this analysis reported only a provisional number
> sourced from `results/summary.json`'s pre-computed `phaseB_*` fields. The user reran
> `src.evaluation.aggregateLODO --arch gru` on the cluster and re-synced; every fold directory now
> has the full set of output files. The table below is read directly from the new
> `runs/lodo_gru/aggregate.json`, not from `summary.json` (whose cached `phaseB_*` fields turned
> out to be a stale snapshot — 0.8474±0.0569 vs. the verified 0.8470±0.0506 — close, but the
> `aggregate.json` next to the actual per-fold runs is the authoritative source and is what's used
> everywhere in this analysis now).

Same winning config, retrained 8 times excluding one WCEB sub-dataset from training each time,
evaluated only on the held-out fold (`runs/lodo_gru/aggregate.json`):

| fold | rouge5 | rouge_l | block_f1 | pages/sec |
|---|---|---|---|---|
| cetd | 0.9202 | 0.9319 | 0.8154 | 2.99 |
| cleaneval | 0.8561 | 0.8877 | 0.8748 | 3.16 |
| cleanportaleval | 0.8292 | 0.8439 | 0.6874 | 3.69 |
| dragnet | 0.8522 | 0.8754 | 0.8107 | 2.47 |
| google-trends-2017 | 0.7402 | 0.7716 | 0.6243 | 1.52 |
| l3s-gn1 | 0.8680 | 0.8693 | 0.7481 | 4.46 |
| readability | 0.8468 | 0.8621 | 0.8237 | 3.07 |
| scrapinghub | 0.8634 | 0.8724 | 0.7878 | 2.97 |
| **mean ± std** | **0.8470 ± 0.0506** | 0.8643 ± 0.0453 | 0.7715 ± 0.0814 | — |

**Phase B (0.8470) is lower than Phase A (0.9068) by 0.060** — confirming the hypothesis from the
earlier provisional numbers: this is the same "Phase B underperforms Phase A" direction as xLSTM
(0.9025→0.8334, a drop of 0.069), not BiLSTM's improvement (0.7183→0.8445, see `bilstm.md`). The
explanation offered there holds up under the verified data: GRU's Phase-A recall was already high
(0.884, see below) from WebMainBench alone, so there's comparatively little for in-domain WCEB
data to fix, while LODO's smaller, narrower per-fold training set (WebMainBench + 7/8 WCEB
datasets, missing the eighth entirely) is a real cost rather than a pure addition. `google-trends
-2017` is again the hardest fold by a wide margin (0.740, block-F1 0.624), consistent with every
other architecture's LODO results.

One new observation the verified data surfaces that wasn't visible in the provisional numbers:
**GRU's LODO throughput (1.52–4.46 pages/sec across folds) is noticeably lower than its Phase-A
throughput (5.23 pages/sec)** — every fold is slower than the Phase-A run, not just the usual
`google-trends-2017` outlier. BiLSTM's and xLSTM's own LODO throughput numbers span similarly wide
ranges (2.58–7.25 and 2.45–5.45 respectively) and are not uniformly below their own Phase-A
numbers, so this looks GRU-specific rather than a general LODO-vs-Phase-A effect. No conclusion is
drawn on the cause: these runs were launched directly in a cluster tmux session rather than
through a resource-managed batch allocation, so there is no record of GPU contention (or lack of
it) to check, and the throughput numbers throughout this analysis should be read as indicative of
relative cost, not as precise, controlled benchmarks.

## Oracle headroom (Phase A)

Oracle ceiling (full 3985-page, gold token-overlap labels, no model): overall ROUGE-5 = 0.9013.
Recomputed locally (`analysis/stats/oracle_vs_gru.md`, full WCEB): **mean diff (oracle − gru) =
−0.0055, 95% CI [−0.0083, −0.0026] — GRU's Phase-A ROUGE-5 is *statistically significantly higher*
than the oracle ceiling overall.** This looks paradoxical (a "ceiling" should not be beatable) but
is explained by what the oracle actually is: gold *token-overlap* keep/drop labels reconstructed
through the same pipeline, not a direct optimization of ROUGE-5 itself — the silver-label heuristic
is an imperfect proxy for "which blocks make the reconstructed text score highest," so a
well-tuned tagger can occasionally make different, ROUGE-5-better choices than the label heuristic
on some pages. Per-dataset, the picture is mixed, not a uniform win: GRU beats oracle
significantly on cetd (mean diff oracle−gru = −0.013) and dragnet (−0.020); oracle beats GRU
significantly on cleaneval (+0.017) and google-trends-2017 (+0.030); the remaining four datasets
(cleanportaleval, l3s-gn1, readability, scrapinghub) show no significant difference either way.
**Practical reading: GRU's Phase-A result has already
closed essentially all of the model-capacity headroom that the oracle ceiling was designed to
measure — remaining differences are dataset-specific and comparable in size to the noise floor of
the silver-labeling heuristic itself**, not a sign that a bigger/different model would score much
higher.

## Caveats

- **Phase B (LODO) is now verified** (2026-08-18) against a fully-populated `runs/lodo_gru/`; the
  earlier provisional number (0.8474±0.0569, sourced only from `summary.json`) is superseded by
  the `aggregate.json`-derived 0.8470±0.0506 used above — very close, but treat `aggregate.json`
  as authoritative if the two ever disagree again.
- GRU's LODO throughput is unexpectedly low relative to its own Phase-A throughput on every fold
  (see the Phase B section). Cause unconfirmed — training was run directly in a cluster tmux
  session, not a resource-managed batch job, so there's no way to check after the fact whether the
  GPU was shared with other work during that run. Treat all pages/sec numbers in this analysis as
  relative-cost indicators, not controlled benchmarks.
- The precision/recall architecture explanation (GRU vs LSTM gating) is a pattern in the data, not
  independently verified by ablation — same caveat as in `bilstm.md`.
- The "GRU beats oracle" result is a real, statistically significant pattern in this data, but it's
  a comment on the label heuristic's imperfection as a proxy, not evidence that GRU has solved
  main-content extraction — see `xlstm.md` for the same phenomenon at smaller magnitude.
