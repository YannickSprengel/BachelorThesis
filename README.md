# Bachelor Thesis

## Topic
HTML to plain text extraction. Strip boilerplate (ads, navigation, footers, cookie banners) and keep only the main content of a web page.

## Goal
Compare extraction quality against compute cost across different tagger architectures: BiLSTM, xLSTM, GRU, and a Transformer. Compare all of them against classical baselines like jusText too.

## Repository Structure
```
src/
  data/            block embedders (MiniLM + hand-crafted structural features)
  caching/         build training caches from WebMainBench and from WCEB
  models/
    lstm/          BiLSTM tagger
    xlstm/         xLSTM tagger
    gru/           GRU tagger
    transformer/   Transformer tagger
    trainCommon.py shared training loop, used by every architecture
    sweep.py       hyperparameter sweep across architectures
  evaluation/      WCEB evaluation, oracle ceiling check, LODO, result aggregation
  baselines/       jusText baseline
docs/
  RESULTS.md       what every output file contains and how to read it
runs/              training runs, created when you train a model
results/           evaluation results, created when you evaluate a model
```

## Setup
Install dependencies with uv:
```
uv sync
```
`mineru-html` is pulled straight from GitHub, not PyPI, so the first sync takes a bit longer.

There is no linter or test suite. The closest thing to a test is:
```
python -m src.models.smokeTest
```
It trains one epoch of every architecture on a small sample cache and checks that nothing crashes.

## Data
Training data comes from WebMainBench, a large collection of pages where the main content is already marked. Most pages are English, a smaller part is Chinese or Japanese.

Evaluation uses WCEB (Bevendorff et al., 2023), a benchmark made of 8 classic English web extraction datasets: cetd, cleaneval, cleanportaleval, dragnet, google-trends-2017, l3s-gn1, readability, and scrapinghub.

Both are turned into block-level training caches. Each page is split into DOM blocks, and each block gets a 431-dim embedding plus a content or boilerplate label. See `src/caching/`.

## Models
Every tagger takes the same input: one 431-dim vector per DOM block (a 384-dim MiniLM embedding plus 47 hand-crafted structural features), and predicts one label per block, content or boilerplate.

- **BiLSTM**: a bidirectional LSTM.
- **xLSTM**: two mLSTM block stacks, one over the sequence forward and one over it reversed.
- **GRU**: a bidirectional GRU. Cheaper than the LSTM at the same size, since it has fewer gates.
- **Transformer**: a small Transformer encoder with full self-attention across all blocks of a page.

All four use the same training loop (`src/models/trainCommon.py`) and the same evaluation code (`src/evaluation/evalCommon.py`), so their results are directly comparable.

## Baselines
- **jusText**: wrapped in `src/baselines/justext.py`.
- **ReaderLM**: not implemented yet. The folder is a placeholder with a git submodule pointing at the original project.

## Experiments
1. Build a training cache from WebMainBench (`src/caching/cacheEmbeddingsForCombined.py`), and from WCEB too if you want leave-one-dataset-out training (`src/caching/cacheEmbeddingsFromWCEB.py`).
2. Train a model, either directly (e.g. `python -m src.models.lstm.trainLSTM`) or through a hyperparameter sweep (`python -m src.models.sweep`).
3. Evaluate on WCEB (`python -m src.evaluation.evaluateBILSTM`, and the same for the other architectures), or run the oracle ceiling check to see the best score any tagger could get with perfect keep/drop decisions (`python -m src.evaluation.evaluateOracleCeiling`).
4. For leave-one-dataset-out cross-validation with WCEB as extra training data, use `src.evaluation.aggregateLODO`.
5. Compare two runs directly with `src.evaluation.compareArchs`, or pull every result into one table with `src.evaluation.aggregateResults`.

Exact commands and flags are in `CLAUDE.md`.

## Results
Every training run writes its own folder under `runs/`. Every evaluation writes a CSV and a JSON file under `results/`. `docs/RESULTS.md` explains what every field in every one of these files means, so you don't have to read the source code to understand a result.
