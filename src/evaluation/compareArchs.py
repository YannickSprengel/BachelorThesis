"""
Paired significance test between two runs' per-page WCEB scores (e.g. two architectures'
sweep winners, or a baseline vs. a scaled-up variant of the same architecture).

run: python -m src.evaluation.compareArchs --a runs/sweep_bilstm/<best>/wceb.csv \
        --b runs/sweep_xlstm/<best>/wceb.csv --metric rouge5 --label-a bilstm --label-b xlstm

Both --a/--b are wceb.csv files written by evalCommon.run_eval (columns: dataset, page_id,
rouge5, rouge_l, n_blocks, n_kept, pred_chars, gt_chars, tp, fp, fn, sec). Rows are joined on
(dataset, page_id), NEVER by row position -- wcebLoader.read_wceb's glob.glob page ordering
is filesystem-dependent, not a guaranteed sort, so two separate eval runs' CSVs aren't safe
to zip row-for-row even though they cover the same WCEB pages.

Uses a paired bootstrap (resample the per-page score differences with replacement, report
the mean difference + 95% CI) instead of adding a scipy dependency for a Wilcoxon test --
numpy is already a hard dependency of this project, scipy is not.

compare_runs() is the reusable core: src.evaluation.ablationScale imports it directly rather
than re-implementing the same comparison for a baseline-vs-scaled pair.
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def load_wceb_csv(path):
    """wceb.csv -> dict[(dataset, page_id)] -> row dict (all columns, as strings)."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[(row["dataset"], row["page_id"])] = row
    return rows


def paired_bootstrap(diffs, n_resamples=10000, seed=42):
    """diffs: per-page (A - B) score differences. Returns mean diff + 95% CI via resampling
    with replacement -- vectorized, ~n_resamples*len(diffs) ints, fine up to ~4000 pages."""
    diffs = np.asarray(diffs, dtype=np.float64)
    n = len(diffs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resample_means = diffs[idx].mean(axis=1)
    return {
        "n_pages": n,
        "mean_diff": float(diffs.mean()),
        "ci_lo": float(np.percentile(resample_means, 2.5)),
        "ci_hi": float(np.percentile(resample_means, 97.5)),
        "frac_resamples_positive": float((resample_means > 0).mean()),
    }


def compare_runs(rows_a, rows_b, metric, n_resamples=10000, seed=42):
    """Core paired comparison, reused by compareArchs.py's CLI and by ablationScale.py."""
    keys_a, keys_b = set(rows_a), set(rows_b)
    common = sorted(keys_a & keys_b)
    if not common:
        raise ValueError("no common (dataset, page_id) pairs between the two wceb.csv files")

    diffs = []
    by_dataset = defaultdict(list)
    for key in common:
        a = float(rows_a[key][metric])
        b = float(rows_b[key][metric])
        diffs.append(a - b)
        by_dataset[key[0]].append(a - b)

    return {
        "metric": metric,
        "n_common": len(common),
        "n_only_a": len(keys_a - keys_b),
        "n_only_b": len(keys_b - keys_a),
        "overall": paired_bootstrap(diffs, n_resamples, seed),
        "by_dataset": {ds: paired_bootstrap(vals, n_resamples, seed)
                       for ds, vals in sorted(by_dataset.items())},
    }


def _is_significant(r):
    return r["ci_lo"] > 0 or r["ci_hi"] < 0


def format_markdown(result, label_a, label_b):
    lines = [f"# Paired comparison: {label_a} vs {label_b}", ""]
    skip_note = ""
    if result["n_only_a"] or result["n_only_b"]:
        skip_note = (f"  (skipped {result['n_only_a']} pages only in A, "
                     f"{result['n_only_b']} only in B)")
    lines.append(f"Metric: `{result['metric']}`  |  common pages: {result['n_common']}{skip_note}")
    lines.append("")
    o = result["overall"]
    verdict = "**significant** (95% CI excludes 0)" if _is_significant(o) else "not significant"
    sign = "A > B" if o["mean_diff"] > 0 else ("A < B" if o["mean_diff"] < 0 else "A = B")
    lines.append(f"**Overall**: mean diff (A-B) = `{o['mean_diff']:+.4f}` ({sign}), "
                 f"95% CI `[{o['ci_lo']:+.4f}, {o['ci_hi']:+.4f}]` -> {verdict}")
    lines.append("")
    lines.append("| dataset | n pages | mean diff (A-B) | 95% CI | significant |")
    lines.append("|---|---|---|---|---|")
    for ds, r in result["by_dataset"].items():
        v = "yes" if _is_significant(r) else "no"
        lines.append(f"| {ds} | {r['n_pages']} | {r['mean_diff']:+.4f} | "
                     f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {v} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="first run's wceb.csv")
    ap.add_argument("--b", required=True, help="second run's wceb.csv")
    ap.add_argument("--label-a", default=None, help="default: --a's path")
    ap.add_argument("--label-b", default=None, help="default: --b's path")
    ap.add_argument("--metric", choices=["rouge5", "rouge_l"], default="rouge5")
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="compare_result", help="output basename for .json/.md")
    args = ap.parse_args()

    rows_a = load_wceb_csv(args.a)
    rows_b = load_wceb_csv(args.b)
    label_a = args.label_a or args.a
    label_b = args.label_b or args.b

    result = compare_runs(rows_a, rows_b, args.metric, args.n_resamples, args.seed)
    result["label_a"], result["label_b"] = label_a, label_b
    result["path_a"], result["path_b"] = args.a, args.b

    md = format_markdown(result, label_a, label_b)
    print(md)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nsaved: {args.out}.json  +  {args.out}.md")


if __name__ == "__main__":
    main()
