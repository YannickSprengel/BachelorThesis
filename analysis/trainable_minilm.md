# Trainable-MiniLM-BiLSTM

> **Labeling note (2026-08-19)**: this architecture's (Phase-A-only, WebMainBench)
> training data used the original word-overlap silver-labeling heuristic
> (`cc-select` word-bag overlap via `blockDataset.py`/`cacheEmbeddingsForCombined.py`),
> found to have real, measured flaws — see `analysis/oracle_investigation.md`. A
> DOM-correspondence relabeling (`--labeler dom`, wired into both
> `cacheEmbeddingsForCombined.py` and `blockDataset.py`/`trainTrainableMiniLM.py`) is
> built and ready to rerun; this document has not yet been updated with those results.

Source data: `results_from_server/runs/20260809-234527_trainable_minilm_bilstm_082e0309/`,
`results_from_server/trainable_minilm/wceb.json`, `analysis/stats/oracle_vs_trainable_minilm.json`,
`analysis/stats/trainable_minilm_vs_bilstm.json`. Field definitions: `docs/RESULTS.md`. Every
number below traces to `analysis/consolidated_data.json` (built by `analysis/collect_data.py`).

This architecture is structurally different from `bilstm.md`/`gru.md`/`xlstm.md` in one specific
way: instead of reading precomputed, frozen 431-dim block embeddings, it fine-tunes the MiniLM
sentence-encoder itself end-to-end, inside the same tagger's forward/backward pass. That
difference is the entire point of this document, so this write-up has no "sweep" section and no
"Phase B" section — neither exists for this architecture, and forcing either would misrepresent
what was actually run. See "What this architecture cannot do" below for why.

## Architecture

`TrainableMiniLMEmbedder` (`src/models/trainableminilm/biLSTMWithTrainableMiniLM.py`): the full
HuggingFace MiniLM encoder, with a hand-rolled forward pass + masked-mean pooling standing in for
`SentenceTransformer.encode()`, because that method is wrapped in `@torch.inference_mode()` and
would block gradients from reaching the encoder weights. Its output is concatenated with the same
47-dim hand-engineered structural feature vector every other architecture uses, then fed into the
**same `BiLSTMTagger` class** (`hidden_dim=128`) the frozen-embedding BiLSTM uses.

Measured parameter count: **118,228,481 total** — embedder 117,653,760 + tagger 574,721. The
tagger head is byte-identical in size to the frozen BiLSTM's; **~99.5% of the extra cost is
fine-tuning MiniLM itself**, not a bigger downstream model. This is the central cost fact for the
rest of this document: everything below compares "same tagger, frozen vs. fine-tuned embedder,"
not two different taggers.

## Training

Exactly **one successful run exists**: `runs/20260809-234527_trainable_minilm_bilstm_082e0309/`,
config `hidden_dim=128, num_layers=1, dropout=0.3, lr=1e-3, epochs=15, min_words=3,
clip_grad_norm=1.0, minilm_chunk_size=null(→16)`. Full 15-epoch curve in `metrics.json`:
`best_epoch=11, best_val_f1=0.8221, early_stopped=false, wall_time_sec=57,358 (≈15.9h)`.

**Reliability**: four earlier attempts at the identical config (hash `68f261a1`) crashed before
writing any progress at all — no `metrics.json`, no `model.pt` — consistent with the git history's
"TrainableMiniLM OOm Fix" commits. Fine-tuning the full encoder is memory-hungry enough that 4/5
attempts failed outright before the chunking/gradient-checkpointing fix landed.

**Cost, concretely**: 15.9 hours to train *this one configuration*. BiLSTM's entire 18-config
sweep (`bilstm.md`) finished in a few hours *combined*. There is no sweep for trainable-MiniLM —
running even a small grid at this per-config cost was not attempted.

## Quality: full WCEB (3985 pages, WebMainBench-trained, zero-leakage guaranteed)

Overall ROUGE-5 F1 = **0.8675**, ROUGE-L F1 = **0.8865**, block-level P/R/F1 = **0.8685 / 0.7114 /
0.7821** (166,993 TP / 25,284 FP / 67,743 FN), throughput **2.891 pages/sec** (0.346s/page mean).

Per-dataset ROUGE-5: cetd 0.931, cleaneval 0.876, cleanportaleval 0.890, dragnet 0.839,
google-trends-2017 0.738 (hardest, consistent with every other architecture), l3s-gn1 0.888,
readability 0.865, scrapinghub 0.853.

**Zero-leakage framing**: unlike every other architecture's Phase A, there is no
WCEB-training-data ingestion path for this architecture at all (no `cacheEmbeddingsFromWCEB.py`
equivalent, no `--exclude-dataset` flag — see "What this architecture cannot do" below). This
result is therefore *structurally guaranteed* to have never trained on any WCEB page, a stronger
leakage guarantee than the other three architectures' own Phase A numbers, which train on
whatever's in `cache/` and could in principle include un-excluded WCEB `.npz` files.

**Recall is the headline change vs. frozen BiLSTM.** BiLSTM's Phase A block-level recall was 0.419
(high precision 0.927, but conservative — missing over half the true content blocks). Fine-tuning
MiniLM instead of freezing it lifts recall to 0.7114 (precision drops to 0.8685) — much closer to
GRU's 0.884 and xLSTM's 0.873 than to frozen BiLSTM's 0.419, though still short of both. Same
tagger class, same loss, same threshold — the only variable that changed between this run and
frozen BiLSTM's Phase A is whether the embedder's weights moved during training, and the recall
gap it closes is the direction "does fine-tuning help" would predict.

### Comparison 1: frozen vs. fine-tuned MiniLM, same tagger (isolates "does fine-tuning help")

| run | embedder | overall ROUGE-5 |
|---|---|---|
| `results/wceb_minilm.json` (2026-06-06) | frozen, 384-dim plain MiniLM | 0.6333 |
| this run | fine-tuned, same MiniLM base | **0.8675** |

**+0.234 ROUGE-5** from fine-tuning alone, same tagger architecture, same base encoder. This is
the cleanest isolation available in this dataset of "does letting the tagging loss update the
embedder help": no significance test was run against this specific legacy baseline (its `wceb.csv`
doesn't exist, only the summary `wceb.json` — see `docs/RESULTS.md` on why the June legacy runs
predate the CSV-per-page schema), but a 0.234-point gap on a 3985-page benchmark is far outside
the noise band every other paired comparison in this analysis has shown (all under 0.04 at the
95% CI). See `figures/frozen_vs_finetuned_minilm.png`.

### Comparison 2: fine-tuned MiniLM vs. frozen combined-features BiLSTM (isolates "is a
fine-tuned single-signal embedder better than a frozen hand-engineered+MiniLM one")

Computed with `compareArchs.py` (paired bootstrap, full 3985-page overlap,
`analysis/stats/trainable_minilm_vs_bilstm.json`): **mean diff = +0.1492, 95% CI [+0.1405,
+0.1581] — significant**, and significant on every one of the 8 WCEB sub-datasets individually
(smallest per-dataset gap: l3s-gn1 +0.099; largest: readability +0.310). Fine-tuning MiniLM beats
the frozen 431-dim hand-engineered+MiniLM combined feature vector, not just the frozen plain
384-dim one.

### Comparison 3: where this lands relative to GRU/xLSTM Phase A and the oracle ceiling

| | ROUGE-5 | ROUGE-L | params | pages/sec |
|---|---|---|---|---|
| bilstm (frozen combined) Phase A | 0.7183 | 0.7422 | 574,721 | 4.28 |
| **trainable-MiniLM (this run)** | **0.8675** | **0.8865** | 118,228,481 | 2.891 |
| gru Phase A | 0.9068 | 0.9226 | 431,105 | 5.23 |
| xlstm Phase A | 0.9025 | 0.9174 | 671,393 | 4.91 |
| oracle ceiling | 0.9013 | 0.9148 | — | — |

Fine-tuning MiniLM closes most, not all, of the gap between frozen BiLSTM and GRU/xLSTM — it lands
about 0.04 ROUGE-5 below both, and 0.034 below the oracle ceiling. That oracle gap is confirmed
with a real CI (`analysis/stats/oracle_vs_trainable_minilm.json`): **mean diff (oracle − model) =
+0.0338, 95% CI [+0.0291, +0.0385], significant overall**. Per-dataset, the pattern already seen
with GRU/xLSTM repeats on one fold: trainable-MiniLM significantly *beats* the oracle on `cetd`
(mean diff −0.0116, CI [−0.0161, −0.0071], oracle behind) while the oracle stays ahead everywhere
else, most by a wide margin on `google-trends-2017` (+0.107). Same conclusion as `overview.md`
Finding 3: the "oracle" here is gold *silver-label* keep/drop decisions, an imperfect proxy for
"which blocks maximize ROUGE-5" — a well-fit model beating it on a specific fold reflects a labeling
artifact on that fold, not the model exceeding a true ceiling.

**Cost side of this trade-off**: trainable-MiniLM is ~274x GRU's parameter count (~206x frozen
BiLSTM's) and its one training run took ~64x GRU's winning-config wall-clock (57,358s vs. 894s),
for a benchmark score still slightly below GRU's. On this benchmark, fine-tuning the embedder is not a Pareto improvement
over choosing a better frozen-embedding architecture — GRU/xLSTM get equal-or-better quality far
more cheaply. See `figures/quality_vs_cost.png` (log-scaled marker size) and
`figures/finetuning_cost.png`.

## What this architecture cannot do

- **No sweep**: not wired into `sweep.py` (deliberate, per its own docstring) — one manually-chosen
  config only. Every comparison above is "this one config vs. the other architectures' *tuned*
  winners," an inherently favorable comparison for the others.
- **No LODO / Phase B**: no WCEB-training-data ingestion path exists for this architecture at all
  (no `cacheEmbeddingsFromWCEB.py` equivalent, no `--exclude-dataset` flag), so a domain-adapted
  number is not just "not yet run" the way GRU's Phase B once was — it is structurally impossible
  with the current code. This is also exactly what makes the zero-leakage guarantee above airtight.

## Caveats

- Single manually-chosen hyperparameter config — there is no sweep to check whether a different
  `lr`/`dropout`/`hidden_dim` would move the ROUGE-5 number meaningfully, so "fine-tuning helps"
  and "we happened to pick reasonable hyperparameters" cannot be cleanly separated here the way
  BiLSTM's sweep separated tuning quality from architecture in `bilstm.md`.
- 4 of 5 total run attempts failed outright (OOM) before the successful one — this architecture is
  markedly less reliable to train than any of the frozen-embedding ones.
- Throughput is the lowest of any architecture evaluated (2.891 vs. 4.28–5.23 pages/sec) since
  inference now runs a full MiniLM forward pass per block instead of a cache lookup — the cost
  shows up at inference time too, not just during training.
- No significance test exists against the frozen plain-MiniLM legacy baseline (Comparison 1) since
  that run predates the per-page `wceb.csv` schema — only the summary-level point estimate is
  available for that specific comparison.
