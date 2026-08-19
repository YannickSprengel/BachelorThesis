# xLSTM

> **Labeling note (2026-08-19)**: Phase A training data here used the original
> word-overlap silver-labeling heuristic (WebMainBench `cc-select` word-bag overlap,
> `cacheEmbeddingsForCombined.py`). Phase B/LODO used WCEB's `overlap_labels`. Both were
> found to have real, measured flaws — see `analysis/oracle_investigation.md`. A
> DOM-correspondence relabeling (Phase A, `--labeler dom`) and a sequential
> position-aware relabeling (Phase B, `--labeler overlap_sequential`) are built and
> ready to rerun; this document has not yet been updated with results from either.

Source data: `results_from_server/runs/sweep_xlstm/`, `results_from_server/runs/lodo_xlstm/`,
`results_from_server/wceb_results.json`, `results_from_server/results/`. Field definitions:
`docs/RESULTS.md`. Every number below traces to `analysis/consolidated_data.json` (built by
`analysis/collect_data.py`).

## Architecture

Not a native bidirectional RNN like BiLSTM/GRU. `in_proj = Linear(431, embedding_dim)` feeds
**two independent mLSTM-only block stacks** (`xLSTMBlockStack`, `slstm_at=[]`): one (`fwd`) runs
on the sequence normally, the other (`bwd`) runs on the sequence reversed (`torch.flip`) and its
output is flipped back before concatenation — two full, independently-parameterized stacks, not
one bidirectional cell. At the sweep-winning config (`embedding_dim=144, num_blocks=2`,
`num_heads=4` fixed) that's **671,393 parameters**, the largest of the three architectures despite
its smaller working dimension, because each mLSTM block carries its own conv1d/qkv projections.
Two training-mechanics differences vs BiLSTM/GRU are baked into its defaults, not just the
architecture: **lower learning rate (5e-4 base, sweep used down to 2.5e-4 for the winner) and
gradient clipping (`clip_grad_norm=1.0`)** — added per `docs/RESULTS.md` as "the fix for the
NaN-blowup instability documented in `git log` (`b138c2c`)". Any quality/convergence difference
from BiLSTM/GRU should be read against this backdrop, not attributed to capacity alone.

## Sweep: hyperparameter search

Grid: `embedding_dim {96,144,192} × num_blocks {1,2,3} × lr {0.00025,0.0005}` = 18 configs.

**Anomaly**: `runs/sweep_xlstm/` contains **36** run directories, not 18 — the full grid was run
twice (2026-07-24/25 and 2026-07-28/29). The first batch's `wceb.*` files were never written
(no top-K eval completed), consistent with an aborted first attempt; only the second batch feeds
`summary.json`/`summary.md` and everything below. Config hashes match 1:1 between batches, so
this is a full rerun of the same grid, not a different one — the first batch is dead weight, not
a second, different experiment.

**Winner**: `embedding_dim=144, num_blocks=2, lr=0.00025` — `val_f1=0.8912`, `wceb_rouge5=0.9025`,
training wall-clock **4045.5s**.

**xLSTM is far more expensive to train than BiLSTM or GRU**: wall-clock across the 18-config grid
ranges 2012–6477s, vs. BiLSTM's 246–939s and GRU's 153–2124s — the *cheapest* xLSTM config in the
grid still takes longer than the *most expensive* BiLSTM config. This is the clearest cost-side
data point in the whole comparison and belongs directly in the quality-vs-cost trade-off section
(`overview.md`), not just this doc.

**Quality is fairly robust to hyperparameters** (though not as flat as GRU's): the five top-K
configs span a 0.0134-wide `wceb_rouge5` band (0.889–0.902), and — notably — the smallest topK
config (`embedding_dim=96, num_blocks=1`, 169,169 params, less than a quarter of the winner's
size) scores 0.895, within 0.008 of the 671K-param winner, while a much larger config
(`embedding_dim=192, num_blocks=3`, 1.5M params) scores *lower* (0.892) than the winner. More
capacity does not reliably buy more quality here either. See `figures/sweep_sensitivity.png`.

## Phase A: WebMainBench → full WCEB (in-domain-trained, cross-domain-evaluated)

Overall ROUGE-5 F1 = **0.9025**, ROUGE-L F1 = **0.9174**, block-level P/R/F1 = **0.939 / 0.873 /
0.905**, throughput 4.91 pages/sec (0.204s/page mean) — essentially tied with GRU on quality
(0.9025 vs 0.9068, see cross-arch significance below) but with more parameters and roughly 6x the
training wall-clock. Recall (0.873) is high like GRU's (0.884), not low like BiLSTM's (0.419) —
same pattern as GRU: neither of the non-LSTM-gate architectures shows BiLSTM's conservative
under-prediction behavior.

Per-dataset ROUGE-5: cetd 0.927, cleaneval 0.896, cleanportaleval 0.912, dragnet 0.900,
google-trends-2017 0.793 (hardest, consistent across every architecture), l3s-gn1 0.915,
readability 0.917, scrapinghub 0.902.

## Comparison to the one legacy xLSTM run

`results_from_server/wceb_results.json` (2026-06-11, `model_combined_xlstm2.pt`, same 671,393
params) predates the sweep: overall ROUGE-5 = **0.6057**, block F1 = 0.682, throughput only
**2.55 pages/sec** (vs. 4.91 now). Two caveats before reading this as a clean before/after: the
legacy run used `keep_threshold=0.3` (not 0.5 — a materially different, manually-chosen operating
point, not just an earlier checkpoint), and the ~2x throughput difference is large enough that
it's worth treating as possibly reflecting implementation/hardware differences from June 2026
rather than pure architecture cost — both ran on the same "Quadro RTX 6000" label, but that alone
doesn't rule out queueing/contention differences on a shared cluster. With that said, the
sweep-tuned Phase A result (0.9025) is **+0.297 ROUGE-5** over this legacy point — by far the
largest legacy-to-current jump of the three architectures (BiLSTM: +0.06 to +0.09, see
`bilstm.md`), consistent with xLSTM having been the least-tuned/newest architecture in June.
See `figures/legacy_vs_current_xlstm.png`.

## Phase B: LODO — domain-adapted, held out one WCEB dataset at a time

Same winning config, retrained 8 times excluding one WCEB sub-dataset from training each time,
evaluated only on the held-out fold (`runs/lodo_xlstm/aggregate.json`):

| fold | rouge5 | rouge_l | block_f1 | pages/sec |
|---|---|---|---|---|
| cetd | 0.9283 | 0.9403 | 0.8547 | 4.12 |
| cleaneval | 0.8255 | 0.8550 | 0.8428 | 5.42 |
| cleanportaleval | 0.8263 | 0.8411 | 0.7008 | 5.45 |
| dragnet | 0.8461 | 0.8682 | 0.8251 | 4.54 |
| google-trends-2017 | 0.7083 | 0.7368 | 0.5844 | 2.45 |
| l3s-gn1 | 0.8872 | 0.8890 | 0.7732 | 5.06 |
| readability | 0.7841 | 0.7965 | 0.8376 | 4.21 |
| scrapinghub | 0.8610 | 0.8694 | 0.7905 | 5.29 |
| **mean ± std** | **0.8334 ± 0.0666** | 0.8495 ± 0.0611 | 0.7761 ± 0.0922 | — |

**Phase B (0.8334) is *lower* than Phase A (0.9025) by 0.069** — the opposite direction from
BiLSTM (which gained +0.126) and roughly the same direction/magnitude as GRU's verified Phase-B
drop (0.9068→0.8470, −0.060, see `gru.md`). Same explanation as offered in `bilstm.md`: xLSTM's Phase-A recall was already high from
WebMainBench alone, so there's less to gain from adding WCEB-domain training data, while LODO's
per-fold training set (WebMainBench + 7/8 WCEB datasets, but excluding the specific held-out
domain entirely) is smaller and narrower than the full training set Phase A used — a real cost, not
just a "no benefit" result. `google-trends-2017` is again the hardest fold, and `readability`
drops much more here (0.784) than it did for BiLSTM's LODO (0.849), worth a closer look if the
thesis wants a per-dataset generalization story rather than just an average.

## Oracle headroom

Oracle ceiling (full 3985-page): overall ROUGE-5 = 0.9013. Recomputed locally
(`analysis/stats/oracle_vs_xlstm.md`, full 3985 pages, replacing the existing partial 263-page
`compare_oracle_vs_xlstm.md`): **mean diff (oracle − xlstm) = −0.0012, 95% CI [−0.0042, +0.0018]
— not significant overall.** Per-dataset it's a mixed picture, same phenomenon as GRU: xLSTM
significantly *beats* oracle on cetd (mean diff oracle−xlstm = −0.008) and dragnet (−0.022);
oracle significantly beats xLSTM on cleaneval (+0.025) and google-trends-2017 (+0.053);
cleanportaleval, l3s-gn1, readability, scrapinghub show no significant difference either way.
**xLSTM's Phase-A result is,
in aggregate, statistically indistinguishable from the oracle ceiling** — reinforcing the same
conclusion as GRU's doc: on Phase A, the remaining gap to "perfect" is now comparable in size to
the noise floor of the silver-labeling heuristic itself, not a sign of remaining model-capacity
headroom.

## Caveats

- The double-sweep-run anomaly (36 dirs, only the second batch usable) — don't accidentally
  aggregate across both batches if re-deriving numbers from `runs/sweep_xlstm/` directly.
- The legacy comparison used a different `keep_threshold` (0.3 vs 0.5) — the +0.297 ROUGE-5
  improvement is real but conflates architecture/tuning improvement with a threshold change.
- Training wall-clock is the standout cost finding for this architecture — make sure it's
  represented in the overview's quality-vs-cost figure with proper weight, since Phase-A quality
  alone makes xLSTM look nearly identical to GRU while cost tells a very different story.
