# Output reference

Every file this pipeline writes, what's in it.

Pipeline order, for context on how these files relate: cache building -> training (produces
`runs/<run_dir>/`) -> single-model WCEB evaluation (produces `wceb.csv`/`wceb.json`, used
standalone by `evaluate*.py` and internally by `sweep.py`/`aggregateLODO.py`) -> sweep
(`runs/sweep_<arch>/`) -> LODO (`runs/lodo_<arch>/`) -> cross-run comparison
(`aggregateResults.py`, `compareArchs.py`, `ablationScale.py`).

---

## 1. Per-run training output — `runs/<timestamp>_<arch>_<confighash>/`

Written by `trainCommon.train()`, called from every `train<ARCH>.py`, `sweep.py`,
`aggregateLODO.py`, and `smokeTest.py`. One directory per training run.

### `config.json`
The exact hyperparameter dict used for this run, written once at the start. Fields vary by
architecture, each architecture has its own hyperparameter space:

| Field | Present for | Meaning |
|---|---|---|
| `arch` | all | `"bilstm"` / `"xlstm"` / `"gru"` / `"transformer"` |
| `lr`, `epochs`, `val_frac`, `seed`, `cache`, `exclude_dataset` | all | shared training knobs; `exclude_dataset` is the LODO held-out fold, `null` otherwise |
| `hidden_dim`, `num_layers`, `dropout` | bilstm, gru | recurrent layer size |
| `embedding_dim`, `num_blocks`, `num_heads`, `context_length`, `clip_grad_norm` | xlstm | block-stack config; `clip_grad_norm=1.0` is the fix for the NaN-blowup instability documented in `git log` (`b138c2c`) |
| `d_model`, `nhead`, `num_layers`, `dim_feedforward`, `dropout`, `clip_grad_norm` | transformer | encoder config; same grad-clip rationale as xLSTM |

### `metrics.json`
Training curve, rewritten after every epoch (so a crash mid-run still leaves the metrics up
to the last completed epoch).
```json
{
  "history": [{"epoch": 0, "train_loss": 0.93, "val_p": 0.94, "val_r": 0.06, "val_f1": 0.12}, ...],
  "best_epoch": 7,
  "best_val_f1": 0.61,
  "early_stopped": false,
  "wall_time_sec": 812.4
}
```
- `history` is one row per epoch, **directly plottable as a training curve** with
  `pandas.DataFrame(json.load(open("metrics.json"))["history"])`, no reshaping needed. Plot
  `val_f1` vs. `epoch` for the classic "did this architecture converge" figure.
- `val_p`/`val_r`/`val_f1` are **block-level** P/R/F1 on the held-out validation split (not
  WCEB ROUGE, that only exists in `wceb.json`, see below). This is the cheap proxy metric
  `sweep.py` ranks configs by, before spending time on a full WCEB pass.
- `wall_time_sec` is wall-clock for the whole training run. Use this for the "training cost"
  side of the quality-vs-cost comparison, alongside `wceb.json`'s inference-time throughput.

### `model.pt`
Raw `state_dict()` of the **best** epoch only (by val F1) — `torch.load("model.pt")` gives you
a dict of tensors, load with `Model().load_state_dict(...)`. **Only written when val F1
improves over the previous best.** If a run never improves past its 0.0 starting point (can
happen in 1-2 epochs on a tiny cache, e.g. during smoke-testing), this file won't exist even
though training completed successfully — check `metrics.json`/`checkpoint.pt` instead if
you need to confirm a run finished.

### `checkpoint.pt`
Full resume state, **overwritten every epoch regardless of improvement** — this is the file
that always exists after any completed epoch, and the correct thing to check for "did this
run at least start training." Contains `epoch`, `model_state`, `optimizer_state`, `best_f1`,
`config`, `history`. Feed its path to `--resume` to continue an interrupted run (needed since
the Gruenau `std` partition caps at 4 days).

---

## 2. Single-model WCEB evaluation — `<out_basename>.csv` / `.json`

Written by `evalCommon.run_eval()`. This is what `evaluate<ARCH>.py --out X` produces
directly, and what `sweep.py`'s top-K stage and `aggregateLODO.py`'s per-fold eval write to
`<run_dir>/wceb.csv`/`.json` internally — same format everywhere.

### `<out>.csv` — one row per WCEB page
| Column | Meaning |
|---|---|
| `dataset` | which of the 8 WCEB sub-datasets (cetd, cleaneval, cleanportaleval, dragnet, google-trends-2017, l3s-gn1, readability, scrapinghub) |
| `page_id` | WCEB's page identifier — join key across different runs' CSVs (see `compareArchs.py`) |
| `rouge5`, `rouge_l` | this page's ROUGE-5 F1 and ROUGE-L F1 against WCEB ground truth |
| `n_blocks`, `n_kept` | total DOM blocks on the page, how many the model kept |
| `pred_chars`, `gt_chars` | reconstructed output length vs. ground-truth length, in characters |
| `tp`, `fp`, `fn` | block-level confusion counts vs. the silver overlap label (not WCEB ground truth directly — see note below) |
| `sec` | extraction time for this one page (simplify + embed + predict + reconstruct only) |

**Important**: `tp`/`fp`/`fn` compare the model's keep/drop decision to a *silver* label
(token-overlap heuristic against WCEB ground truth), not to WCEB ground truth text directly
— that's what `rouge5`/`rouge_l` measure. The two can disagree on individual pages; both are
legitimate, they're just different questions ("did I pick the right blocks" vs. "does the
resulting text match").

### `<out>.json` — aggregate summary
```json
{
  "model": "runs/.../model.pt", "arch": "bilstm",
  "device": "cuda", "hardware": "Quadro RTX 6000", "n_params": 574721,
  "keep_threshold": 0.5, "label_threshold": 0.5,
  "rouge5": {"n": 5, "n_docs": 3985, "n_skipped": 0, "overall_f1": 0.69,
             "by_dataset": {"cetd": {"n": 700, "f1": 0.75}, ...}},
  "rouge_l": {"n_docs": 3985, "n_skipped": 0, "overall_f1": 0.70, "by_dataset": {...}},
  "block_level": {"note": "silver labels (token overlap with GT text)",
                  "precision": 0.88, "recall": 0.36, "f1": 0.51,
                  "tp": 12000, "fp": 1600, "fn": 21000, "by_dataset": {"cetd": {"f1": 0.6}, ...}},
  "throughput": {"pages": 3985, "total_extract_sec": 870.5, "pages_per_sec": 4.58,
                 "sec_per_page_mean": 0.218, "sec_per_page_median": 0.19}
}
```
- `n_params` is the number this pipeline uses everywhere else for "model size" (e.g. in
  `aggregateResults.py`'s table) — it's a live count from the loaded model, not hand-entered.
- `keep_threshold` is the model's sigmoid keep/drop cutoff (the knob worth tuning per model);
  `label_threshold` is the silver-label overlap cutoff (keep this fixed when comparing
  different `keep_threshold` values, otherwise you're comparing against a moving target).
- `throughput.pages_per_sec` is the **inference-cost** side of quality-vs-cost — pair with
  `n_params` and `metrics.json`'s `wall_time_sec` (training cost) for the full cost picture.
- `by_dataset` breakdowns exist for both ROUGE metrics and block-level F1 — use these for a
  per-dataset bar chart, not just the overall number, since datasets vary a lot in difficulty
  (readability is consistently the hardest across every model tried so far).

---

## 3. Oracle ceiling — `results/oracle/wceb.csv` / `.json`

Written by `evaluateOracleCeiling.py`. **Same CSV shape as above minus `tp`/`fp`/`fn`/`sec`**
(no model, so no block-level P/R/F1 vs. its own labels, and no timing). JSON is the same
`rouge5`/`rouge_l` blocks, plus `"keep_source": "oracle (gold token-overlap labels, no
model)"` instead of `model`/`arch`/`n_params`/`device`/etc.

**Thesis use**: this is the ceiling line on every quality chart — no trained model can score
above this, since it assumes perfect keep/drop decisions under the same reconstruction
pipeline and label heuristic every other number in this repo uses. The gap between a trained
model's `overall_f1` and this file's `overall_f1` is the headroom actually available to
model/data improvements (as opposed to headroom that's really a pipeline/labeling limit).

---

## 4. Sweep leaderboard — `runs/sweep_<arch>/summary.csv` / `.json`

Written by `sweep.py`. One row per hyperparameter config tried.

| Column | Present for | Meaning |
|---|---|---|
| swept hyperparam columns (e.g. `hidden_dim`, `dropout`, `lr`) | all rows | whatever `--grid` keys were varied |
| `run_dir` | all rows | points to that config's full `runs/<run_dir>/` (config.json/metrics.json/model.pt) |
| `val_p`, `val_r`, `val_f1` | all rows | best-epoch validation block-F1 (the cheap ranking metric) |
| `wall_time_sec` | all rows | this config's training wall-clock |
| `wceb_rouge5`, `wceb_rouge_l`, `wceb_block_f1`, `wceb_pages_per_sec`, `wceb_n_params` | **top-K rows only** | full WCEB eval — **blank, not zero, for every other row**. A blank here means "not evaluated," not "scored zero." |

Rows are sorted by `val_f1` descending, so row 1..K (whatever `--topk` was) are the rows with
`wceb_*` populated.

**Thesis use**: this is your hyperparameter-sensitivity table/plot (val_f1 vs. each
hyperparameter). The `wceb_*` columns on the top rows are what feeds `aggregateResults.py`.

**Checking progress on a running sweep**: `summary.csv` is written as soon as the val-F1
leaderboard stage finishes (with `wceb_*` columns still blank), then rewritten again after
*every* top-K config's full WCEB eval, not just once at the very end. The top-K stage is the
slow part of a sweep, a full 3985-page WCEB pass per config, easily hours for `--topk 5`. If a
sweep looks stalled, check whether this file exists yet and whether its `wceb_*` columns are
filling in one row at a time, that's a more reliable progress signal than watching console
output on a SLURM job (which can sit buffered and invisible for a while even though the run is
progressing normally).

---

## 5. LODO aggregate — `runs/lodo_<arch>/aggregate.json`

Written by `aggregateLODO.py`. One fixed hyperparameter config, retrained 8 times (once per
held-out WCEB sub-dataset).
```json
{
  "arch": "bilstm", "config": "runs/sweep_bilstm/.../config.json",
  "folds": ["cetd", "cleaneval", ...],
  "per_fold": [{"fold": "cetd", "val_f1": 0.6, "rouge5": 0.71, "rouge_l": 0.73,
                "block_f1": 0.58, "n_params": 574721, "pages_per_sec": 4.4,
                "run_dir": "runs/lodo_bilstm/lodo_bilstm_cetd"}, ...],
  "rouge5": {"mean": 0.70, "std": 0.03},
  "rouge_l": {"mean": 0.72, "std": 0.02},
  "block_f1": {"mean": 0.59, "std": 0.04}
}
```
- `per_fold[i]`'s `rouge5`/`rouge_l`/`block_f1` are that fold's score **evaluated only on the
  held-out dataset** (`fold`), from a model that never saw that dataset in training — this is
  the genuinely held-out number, not a WCEB-wide average.
- The top-level `rouge5`/`rouge_l`/`block_f1` blocks are mean±std **across the 8 folds** —
  this is the number to compare against Phase A's single WCEB-wide `overall_f1` (see below,
  Phase A trained with zero WCEB data, Phase B added 7/8 WCEB datasets per fold).
- `n_params`/`pages_per_sec` are duplicated per fold (same fixed config every fold) purely so
  `per_fold` is self-contained — read `per_fold[0]` if you need them, they don't vary.

**Thesis use**: mean±std across folds is your "does adding matching-domain training data
help, and how much does it vary by which dataset was held out" number — report both, the std
tells you how sensitive the result is to which fold, not just the average effect.

---

## 6. Consolidated comparison — `results/summary.csv` / `.json` / `.md`

Written by `aggregateResults.py`. One row per architecture, everything above lined up side by
side: oracle ceiling, best Phase-A (WebMainBench-only) result, Phase-B (LODO) mean±std. The
`.md` file is a ready-to-paste Markdown table for the thesis draft; `.csv`/`.json` have the
full field set (see the script's own row-building code for the exact key list, or just open
the `.json` — it's the same `phaseA_*`/`phaseB_*`/`oracle_*` prefixed keys described in
sections 2-5 above, just gathered onto one row per architecture). A missing `oracle_*` or
`phaseB_*` column for a given architecture means that data hasn't been produced yet
(oracle not run, or LODO not run for that architecture) — not that it scored zero or failed.

---

## 7. Pairwise significance — `<out>.json` / `.md` (from `compareArchs.py`)

Written by `compareArchs.py`, and reused (same function, `compare_runs()`) by
`ablationScale.py` below.
```json
{
  "metric": "rouge5", "n_common": 3985, "n_only_a": 0, "n_only_b": 0,
  "overall": {"n_pages": 3985, "mean_diff": -0.21, "ci_lo": -0.22, "ci_hi": -0.20,
              "frac_resamples_positive": 0.0},
  "by_dataset": {"cetd": {"n_pages": 700, "mean_diff": -0.19, ...}, ...},
  "label_a": "bilstm", "label_b": "oracle", "path_a": "...", "path_b": "..."
}
```
- `mean_diff` is **A minus B** per page, averaged — positive means A scored higher.
- `ci_lo`/`ci_hi` are the 95% bootstrap confidence interval on that mean difference.
  **"Significant" here means the CI excludes 0** — printed explicitly in the `.md` report, no
  need to eyeball the numbers.
- `n_only_a`/`n_only_b`: pages present in one run's CSV but not the other's (shouldn't happen
  comparing two full-WCEB runs, but can if one run used `--datasets` to only cover a subset —
  those pages are silently excluded from the comparison, not treated as a zero score).

**Thesis use**: this is what turns "architecture X scored 0.02 higher than Y" into a claim you
can actually defend — report the CI, not just the point estimate, whenever comparing two
architectures or two configs.

---

## 8. Smoke test — `runs/smoke_test/<arch>/` + `runs/smoke_test/smoke_test.json`

Written by `smokeTest.py`. Each `<arch>/` is a normal 2-epoch training run dir (section 1
above). `smoke_test.json` is a list of per-architecture pass/fail records:
```json
[{"arch": "bilstm", "status": "PASS", "train_loss": 0.78, "val_f1": 0.35, "sec": 0.3},
 {"arch": "transformer", "status": "FAIL", "reason": "RuntimeError: ...", "sec": 0.2}]
```
Not a thesis artifact — this is a development/CI check confirming the pipeline runs before
spending cluster time. Only worth mentioning in the thesis methodology section as "an
integration check exists," not as a results table.

---

## 9. Scaled-parameter ablation — `runs/ablation_scale_bilstm/report.json` / `.md`

Written by `ablationScale.py`. Baseline BiLSTM (`hidden_dim=128`) vs. a scaled-up BiLSTM
(bigger `hidden_dim`, same cache/split/seed/epochs/lr) trained and evaluated identically, plus
the `compareArchs.compare_runs()` significance result between them. `baseline/` and `scaled/`
subdirectories are normal run dirs (section 1). `report.json`/`.md` hold the side-by-side
table: param count, ROUGE-5/L, block P/R/F1, throughput, and the bootstrap CI on the
ROUGE-5 difference — see `ablationScale.py`'s own docstring for the exact field list, kept
next to the code since this is the most recently added script and most likely to change.

**Thesis use**: this directly answers "does more capacity on the best-understood architecture
already close some of the oracle-ceiling gap, or does it take a different architecture" —
the headline number for this section is the bootstrap CI on the ROUGE-5 delta, not just the
raw before/after point estimate.

---

## Naming conventions used consistently across all the files above

- `_f1` / `_p` / `_r` suffixes always mean the metric they're attached to (`val_f1` = block
  F1 on the validation split, `rouge5`/`rouge_l` are already full metric names).
- `_sec` suffix always means seconds; `_mean`/`_std`/`_lo`/`_hi` always refer to the
  aggregation applied, not a different metric.
- `n_*` prefix always means a count (`n_docs`, `n_params`, `n_blocks`, `n_kept`, `n_common`).
- A missing/blank field in a CSV or a missing key in a JSON dict means "not computed for this
  row" (e.g. a non-top-K sweep row, or oracle/LODO not run yet) — **never** means zero. Check
  for presence before treating an absent value as 0 in any thesis figure or table.
