"""
Grid/random hyperparameter sweep with tracking, for BiLSTM or xLSTM. Loads the cache once,
splits train/val once, trains every config, ranks by (cheap) val block-F1, then runs the
expensive full-WCEB eval only on the top-K configs.

run:
  python -m src.models.sweep --arch bilstm --cache cache/ \
      --wceb src/evaluation/web-content-extraction-benchmark/datasets/combined \
      --grid hidden_dim=64,128,256 --grid dropout=0.1,0.3,0.5 --grid lr=1e-3,5e-4 \
      --mode grid --epochs 15 --patience 4 --topk 5 --out runs/sweep_bilstm_20260724

No Optuna/Bayesian search on purpose: this is a plain grid/random driver sized for a
bachelor thesis (~10-30 configs), not a production ML platform.
"""

import argparse
import csv
import itertools
import json
import os
import random

import torch

from src.models import trainCommon
from src.evaluation.evalCommon import run_eval
from src.models.lstm.trainLSTM import build_model as build_bilstm
from src.models.xlstm.trainxLSTM import build_model as build_xlstm
from src.models.gru.trainGRU import build_model as build_gru
from src.models.transformer.trainTransformer import build_model as build_transformer

ARCH_BUILDERS = {
    "bilstm": build_bilstm, "xlstm": build_xlstm,
    "gru": build_gru, "transformer": build_transformer,
}
ARCH_DEFAULTS = {
    "bilstm": {"lr": 1e-3, "clip_grad_norm": None},
    "xlstm": {"lr": 5e-4, "clip_grad_norm": 1.0},
    "gru": {"lr": 1e-3, "clip_grad_norm": None},
    "transformer": {"lr": 5e-4, "clip_grad_norm": 1.0},
}


def _parse_value(s):
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def parse_grid(grid_args):
    grid = {}
    for item in grid_args:
        key, _, vals = item.partition("=")
        grid[key] = [_parse_value(v) for v in vals.split(",")]
    return grid


def grid_configs(grid):
    if not grid:
        return [{}]
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def random_configs(grid, n_samples, seed):
    if not grid:
        return [{}]
    rng = random.Random(seed)
    keys = list(grid.keys())
    return [{k: rng.choice(grid[k]) for k in keys} for _ in range(n_samples)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(ARCH_BUILDERS))
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--wceb", required=True, help="path to .../datasets/combined")
    ap.add_argument("--grid", action="append", default=[], help="key=v1,v2,... (repeatable)")
    ap.add_argument("--mode", choices=["grid", "random"], default="grid")
    ap.add_argument("--n-samples", type=int, default=10, help="config count for --mode random")
    ap.add_argument("--max-configs", type=int, default=0, help="0 = no cap")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-dataset", default=None,
                     help="drop wceb-<name>-*.npz cache files for leave-one-dataset-out training")
    ap.add_argument("--topk", type=int, default=5, help="how many configs get a full WCEB eval")
    ap.add_argument("--out", required=True, help="output dir for run subdirs + summary.csv/json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    build_model = ARCH_BUILDERS[args.arch]
    arch_defaults = ARCH_DEFAULTS[args.arch]

    files = trainCommon.list_cache_files(args.cache, exclude_dataset=args.exclude_dataset)
    train_files, val_files = trainCommon.train_val_split(files, args.val_frac, args.seed)
    print(f"train={len(train_files)}  val={len(val_files)}  device={device}", flush=True)
    train_data = trainCommon.load_cache_to_memory(train_files)
    val_data = trainCommon.load_cache_to_memory(val_files)

    grid = parse_grid(args.grid)
    combos = grid_configs(grid) if args.mode == "grid" else random_configs(grid, args.n_samples, args.seed)
    if args.max_configs:
        combos = combos[:args.max_configs]
    print(f"sweeping {len(combos)} configs ({args.mode})", flush=True)

    os.makedirs(args.out, exist_ok=True)
    leaderboard = []
    for i, combo in enumerate(combos):
        config = {
            **arch_defaults, **combo, "arch": args.arch, "epochs": args.epochs,
            "val_frac": args.val_frac, "seed": args.seed, "cache": args.cache,
            "exclude_dataset": args.exclude_dataset,
        }
        print(f"\n--- config {i + 1}/{len(combos)}: {combo} ---", flush=True)
        model = build_model(config)
        run_dir = trainCommon.make_run_dir(args.out, args.arch, config)
        metrics = trainCommon.train(
            config, model, train_data, val_data, device, run_dir,
            epochs=args.epochs, lr=config["lr"], patience=args.patience,
            min_delta=args.min_delta, clip_grad_norm=config.get("clip_grad_norm"),
        )
        best_entry = next(h for h in metrics["history"] if h["epoch"] == metrics["best_epoch"])
        leaderboard.append({
            **combo, "run_dir": run_dir,
            "val_p": round(best_entry["val_p"], 4), "val_r": round(best_entry["val_r"], 4),
            "val_f1": round(metrics["best_val_f1"], 4),
            "wall_time_sec": round(metrics["wall_time_sec"], 1),
        })

    leaderboard.sort(key=lambda r: r["val_f1"], reverse=True)
    print("\n=== Leaderboard (by val F1) ===", flush=True)
    for rank, row in enumerate(leaderboard, 1):
        print(f"{rank:2d}. val_f1={row['val_f1']:.4f}  {row['run_dir']}", flush=True)

    summary_csv = os.path.join(args.out, "summary.csv")
    summary_json = os.path.join(args.out, "summary.json")

    def write_summary():
        # Rewritten after every topK config, not just once at the end: the topK stage below
        # (a full WCEB pass per config) is the slowest part of a sweep and can run for hours
        # with very little console output, so checking whether this file has grown is a much
        # more reliable "is this still working" signal than watching a possibly-buffered log.
        all_keys = sorted({k for row in leaderboard for k in row.keys()})
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for row in leaderboard:
                writer.writerow(row)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(leaderboard, f, indent=2)

    write_summary()
    print(f"\nsaved (val-F1 only so far): {summary_csv}  +  {summary_json}", flush=True)

    topk = leaderboard[:args.topk]
    print(f"\nRunning full WCEB eval on top {len(topk)} configs "
          f"(the slow part -- {summary_csv} updates after each one finishes)...", flush=True)
    for i, row in enumerate(topk):
        run_dir = row["run_dir"]
        print(f"\n[topK {i + 1}/{len(topk)}] evaluating {run_dir} on full WCEB...", flush=True)
        with open(os.path.join(run_dir, "config.json")) as f:
            config = json.load(f)
        model = build_model(config).to(device)
        model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location=device))
        model.eval()
        summary = run_eval(model, args.wceb, device=device, model_path=os.path.join(run_dir, "model.pt"),
                            arch=args.arch, out_basename=os.path.join(run_dir, "wceb"))
        row["wceb_rouge5"] = summary["rouge5"]["overall_f1"]
        row["wceb_rouge_l"] = summary["rouge_l"]["overall_f1"]
        row["wceb_block_f1"] = summary["block_level"]["f1"]
        row["wceb_pages_per_sec"] = summary["throughput"]["pages_per_sec"]
        row["wceb_n_params"] = summary["n_params"]
        print(f"[topK {i + 1}/{len(topk)}] {run_dir}: rouge5={row['wceb_rouge5']:.4f}  "
              f"rougeL={row['wceb_rouge_l']:.4f}  block_f1={row['wceb_block_f1']:.4f}  "
              f"pages/sec={row['wceb_pages_per_sec']:.2f}  n_params={row['wceb_n_params']:,}",
              flush=True)
        write_summary()

    print(f"\nsaved: {summary_csv}  +  {summary_json}", flush=True)


if __name__ == "__main__":
    main()
