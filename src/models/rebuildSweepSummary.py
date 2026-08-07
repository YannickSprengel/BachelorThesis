"""
Rebuilds summary.csv/json for a sweep.py run that crashed (or was killed by a job time limit)
before writing them. Scans --out for run subdirs that already have config.json + metrics.json +
model.pt from a completed training pass, reconstructs the leaderboard from those files, then
runs the same topK full-WCEB eval sweep.py's tail does. No retraining involved.

run:
  python -m src.models.rebuildSweepSummary --arch bilstm \
      --wceb src/evaluation/web-content-extraction-benchmark/datasets/combined \
      --topk 5 --out runs/sweep_bilstm_20260728...
"""

import argparse
import glob
import json
import os

import torch

from src.models.sweep import ARCH_BUILDERS, write_summary, run_topk_eval

_SKIP_KEYS = {"arch", "epochs", "val_frac", "seed", "cache", "exclude_dataset"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(ARCH_BUILDERS))
    ap.add_argument("--wceb", required=True, help="path to .../datasets/combined")
    ap.add_argument("--topk", type=int, default=5, help="how many configs get a full WCEB eval")
    ap.add_argument("--out", required=True, help="sweep dir containing already-trained run subdirs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    build_model = ARCH_BUILDERS[args.arch]

    run_dirs = sorted(d for d in glob.glob(os.path.join(args.out, "*")) if os.path.isdir(d))
    leaderboard = []
    for run_dir in run_dirs:
        config_path = os.path.join(run_dir, "config.json")
        metrics_path = os.path.join(run_dir, "metrics.json")
        model_path = os.path.join(run_dir, "model.pt")
        if not (os.path.exists(config_path) and os.path.exists(metrics_path) and os.path.exists(model_path)):
            print(f"skip {run_dir}: missing config.json/metrics.json/model.pt", flush=True)
            continue
        with open(config_path) as f:
            config = json.load(f)
        with open(metrics_path) as f:
            metrics = json.load(f)
        best_entry = next(h for h in metrics["history"] if h["epoch"] == metrics["best_epoch"])
        combo = {k: v for k, v in config.items() if k not in _SKIP_KEYS}
        leaderboard.append({
            **combo, "run_dir": run_dir,
            "val_p": round(best_entry["val_p"], 4), "val_r": round(best_entry["val_r"], 4),
            "val_f1": round(metrics["best_val_f1"], 4),
            "wall_time_sec": round(metrics["wall_time_sec"], 1),
        })

    leaderboard.sort(key=lambda r: r["val_f1"], reverse=True)
    print(f"\nfound {len(leaderboard)} completed runs under {args.out}", flush=True)
    for rank, row in enumerate(leaderboard, 1):
        print(f"{rank:2d}. val_f1={row['val_f1']:.4f}  {row['run_dir']}", flush=True)

    summary_csv, summary_json = write_summary(leaderboard, args.out)
    print(f"\nsaved (val-F1 only so far): {summary_csv}  +  {summary_json}", flush=True)

    run_topk_eval(build_model, leaderboard, args.topk, args.out, args.wceb, args.arch, device)

    print(f"\nsaved: {summary_csv}  +  {summary_json}", flush=True)


if __name__ == "__main__":
    main()