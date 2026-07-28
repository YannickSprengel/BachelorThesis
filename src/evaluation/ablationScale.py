"""
Controlled ablation: train a baseline BiLSTM (hidden_dim=128, the existing default) and a
scaled-up BiLSTM (hidden_dim multiplied up, default 512 = 4x) that is IDENTICAL in every
other respect -- same cache, same train/val split (same seed), same epochs, same lr, same
architecture family (still BiLSTMTagger / nn.LSTM) -- so any difference in the results is
attributable to parameter count alone, not a confounded architecture or data change.

Reuses trainCommon.train() + evalCommon.run_eval() (the same "train then run_eval" pattern
sweep.py and aggregateLODO.py use) and compareArchs.compare_runs()/format_markdown() for the
significance test between the two runs' per-page WCEB ROUGE-5 scores, rather than
reimplementing any of it.

run: python -m src.evaluation.ablationScale --cache cache/ \
        --wceb src/evaluation/web-content-extraction-benchmark/datasets/combined \
        --out runs/ablation_scale_bilstm

Output: runs/ablation_scale_bilstm/{baseline/, scaled/, report.json, report.md}
  baseline/, scaled/ -- normal training run dirs (config.json/model.pt/checkpoint.pt/
    metrics.json/wceb.csv/wceb.json), see docs/RESULTS.md sections 1-2.
  report.json/.md -- side-by-side table (param count, ROUGE-5/L, block P/R/F1, throughput)
    plus the bootstrap significance result on the ROUGE-5 delta -- see docs/RESULTS.md
    section 9 for the field reference.
"""

import argparse
import json
import os

import torch

from src.models import trainCommon
from src.evaluation.evalCommon import run_eval
from src.evaluation.compareArchs import load_wceb_csv, compare_runs, format_markdown
from src.models.lstm.trainLSTM import build_model


def train_and_eval(label, config, train_data, val_data, wceb_dir, epochs, lr, out_dir, device):
    model = build_model(config)
    run_dir = os.path.join(out_dir, label)
    print(f"\n=== training {label} (hidden_dim={config['hidden_dim']}) -> {run_dir} ===")
    trainCommon.train(config, model, train_data, val_data, device, run_dir,
                       epochs=epochs, lr=lr, clip_grad_norm=None)

    eval_model = build_model(config).to(device)
    eval_model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location=device))
    eval_model.eval()
    print(f"=== evaluating {label} on full WCEB ===")
    summary = run_eval(eval_model, wceb_dir, device=device,
                        model_path=os.path.join(run_dir, "model.pt"), arch="bilstm",
                        out_basename=os.path.join(run_dir, "wceb"))
    return run_dir, summary


def row_summary(summary):
    return {
        "n_params": summary["n_params"],
        "rouge5": summary["rouge5"]["overall_f1"],
        "rouge_l": summary["rouge_l"]["overall_f1"],
        "block_precision": summary["block_level"]["precision"],
        "block_recall": summary["block_level"]["recall"],
        "block_f1": summary["block_level"]["f1"],
        "pages_per_sec": summary["throughput"]["pages_per_sec"],
    }


def format_report_md(report):
    b, s = report["baseline"], report["scaled"]
    scale_factor = s["n_params"] / b["n_params"] if b["n_params"] else float("nan")
    lines = ["# Scaled-parameter BiLSTM ablation", ""]
    lines.append(f"Baseline `hidden_dim={b['hidden_dim']}` ({b['n_params']:,} params) vs. "
                 f"scaled `hidden_dim={s['hidden_dim']}` ({s['n_params']:,} params, "
                 f"{scale_factor:.1f}x). Identical cache, split, seed, epochs, lr -- the only "
                 f"thing that differs is parameter count.")
    lines.append("")
    lines.append("| metric | baseline | scaled | delta |")
    lines.append("|---|---|---|---|")
    rows = [
        ("n_params", "params", True), ("rouge5", "ROUGE-5", False), ("rouge_l", "ROUGE-L", False),
        ("block_precision", "block P", False), ("block_recall", "block R", False),
        ("block_f1", "block F1", False), ("pages_per_sec", "pages/sec", False),
    ]
    for key, label, is_int in rows:
        bv, sv = b[key], s[key]
        if is_int:
            lines.append(f"| {label} | {bv:,} | {sv:,} | {sv - bv:+,} |")
        else:
            lines.append(f"| {label} | {bv:.4f} | {sv:.4f} | {sv - bv:+.4f} |")
    lines.append("")
    lines.append("## ROUGE-5 significance (paired bootstrap, scaled vs. baseline)")
    lines.append("")
    lines.append(format_markdown(report["rouge5_significance"], "scaled", "baseline"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--wceb", required=True, help="path to .../datasets/combined")
    ap.add_argument("--baseline-hidden-dim", type=int, default=128)
    ap.add_argument("--scale-hidden-dim", type=int, default=512,
                     help="scaled variant's hidden_dim (default 4x the baseline)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/ablation_scale_bilstm")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # loaded ONCE, reused for both runs -- same reason sweep.py preloads the cache: avoids
    # re-reading disk per run and, more importantly here, GUARANTEES both runs see the exact
    # same train/val split (same seed applied to the same file list).
    files = trainCommon.list_cache_files(args.cache)
    train_files, val_files = trainCommon.train_val_split(files, args.val_frac, args.seed)
    print(f"train={len(train_files)}  val={len(val_files)}  device={device}")
    train_data = trainCommon.load_cache_to_memory(train_files)
    val_data = trainCommon.load_cache_to_memory(val_files)

    base_cfg = {
        "arch": "bilstm", "hidden_dim": args.baseline_hidden_dim, "num_layers": 1,
        "dropout": 0.3, "lr": args.lr, "epochs": args.epochs, "val_frac": args.val_frac,
        "seed": args.seed, "cache": args.cache, "exclude_dataset": None,
    }
    scaled_cfg = {**base_cfg, "hidden_dim": args.scale_hidden_dim}

    baseline_dir, baseline_summary = train_and_eval(
        "baseline", base_cfg, train_data, val_data, args.wceb, args.epochs, args.lr, args.out, device)
    scaled_dir, scaled_summary = train_and_eval(
        "scaled", scaled_cfg, train_data, val_data, args.wceb, args.epochs, args.lr, args.out, device)

    rows_scaled = load_wceb_csv(os.path.join(scaled_dir, "wceb.csv"))
    rows_baseline = load_wceb_csv(os.path.join(baseline_dir, "wceb.csv"))
    significance = compare_runs(rows_scaled, rows_baseline, metric="rouge5")
    significance["label_a"], significance["label_b"] = "scaled", "baseline"

    report = {
        "baseline": {"hidden_dim": args.baseline_hidden_dim, "run_dir": baseline_dir,
                     **row_summary(baseline_summary)},
        "scaled": {"hidden_dim": args.scale_hidden_dim, "run_dir": scaled_dir,
                   **row_summary(scaled_summary)},
        "rouge5_significance": significance,
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    md = format_report_md(report)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print("\n" + md)
    print(f"\nsaved: {args.out}/report.json  +  {args.out}/report.md")


if __name__ == "__main__":
    main()
