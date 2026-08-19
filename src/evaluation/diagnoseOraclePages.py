"""
Per-page diagnostic features for the oracle-ceiling investigation (see
analysis/oracle_investigation.md). Two things share the same plumbing so they live in one
script rather than two:

1. Bare-cell hypothesis test (--pages all): for every WCEB page, compute what fraction of
   kept blocks are bare td/th/li/dt/dd (table cells / list items that lost their
   table/list wrapper during block segmentation -- see simplify_html's
   include_parents=False path). Joins rouge5/n_blocks in from an existing oracle wceb.csv
   rather than recomputing reconstruct()/to_text()/rouge, so this pass only needs
   parse_page + block_texts + overlap_labels -- no reconstruction, cheap over all pages.
2. Worst-tail forensics (--pages worst): the same features restricted to the pages an
   existing oracle run scored below --worst-threshold, plus optional --dump-diff for a
   handful of representative pages.

run: python -m src.evaluation.diagnoseOraclePages --wceb src/evaluation/web-content-extraction-benchmark/datasets/combined \
        --oracle-csv results/oracle/wceb.csv --pages all --out results/oracle_diagnosis/bare_cell_features \
        --stats-out analysis/stats/bare_cell_hypothesis
"""

import argparse
import csv
import difflib
import json
import os
from collections import defaultdict

import numpy as np

from src.evaluation.wcebLoader import read_wceb
from src.evaluation.blockReconstruction import parse_page, block_texts, overlap_labels
from src.evaluation.textMetrics import tokenize

BARE_TAGS = {"td", "th", "li", "dt", "dd"}


def load_oracle_csv(path):
    """oracle wceb.csv -> dict[(dataset, page_id)] -> row dict."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[(row["dataset"], row["page_id"])] = row
    return rows


def page_features(simpl_blocks, keep):
    """Cheap per-page diagnostic features from the labeling decision alone -- no
    reconstruct()/to_text()/rouge needed, rouge5/n_blocks are joined in from an existing
    oracle run instead."""
    kept = [b for b, k in zip(simpl_blocks, keep) if k]
    n_bare_kept = sum(1 for b in kept if b.name in BARE_TAGS)
    n_bare_total = sum(1 for b in simpl_blocks if b.name in BARE_TAGS)
    n_truncated_kept = sum(1 for b in kept if "..." in b.get_text())
    return {
        "n_kept": len(kept),
        "bare_cell_frac": n_bare_kept / len(kept) if kept else 0.0,
        "has_bare_cell": n_bare_kept > 0,
        "n_bare_total": n_bare_total,
        "truncated_kept_frac": n_truncated_kept / len(kept) if kept else 0.0,
    }


def unpaired_bootstrap(group_a, group_b, n_resamples=10000, seed=42):
    """Two independent samples (NOT the same pages under two conditions, unlike
    compareArchs.paired_bootstrap) -- resample each group independently with
    replacement, report the difference of resampled means + 95% CI. Same
    numpy-vectorized shape as paired_bootstrap."""
    a, b = np.asarray(group_a, dtype=np.float64), np.asarray(group_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(a), size=(n_resamples, len(a))) if len(a) else None
    idx_b = rng.integers(0, len(b), size=(n_resamples, len(b))) if len(b) else None
    if idx_a is None or idx_b is None:
        return {"n_a": len(a), "n_b": len(b), "mean_diff": None, "ci_lo": None, "ci_hi": None}
    resample_diffs = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    return {
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(a.mean() - b.mean()),
        "ci_lo": float(np.percentile(resample_diffs, 2.5)),
        "ci_hi": float(np.percentile(resample_diffs, 97.5)),
    }


def _is_significant(r):
    return r["ci_lo"] is not None and (r["ci_lo"] > 0 or r["ci_hi"] < 0)


def format_bare_cell_markdown(by_dataset, overall):
    lines = ["# Bare-cell hypothesis: has-bare-cell vs. no-bare-cell pages, by dataset", ""]
    lines.append("`bare cell` = a kept block whose own tag is td/th/li/dt/dd (lost its "
                  "table/list wrapper during segmentation -- see analysis/oracle_investigation.md).")
    lines.append("")
    v = "yes" if _is_significant(overall) else "no"
    lines.append(f"**Overall**: has-bare-cell mean rouge5={overall['mean_a']:.4f} (n={overall['n_a']}) "
                 f"vs. no-bare-cell mean={overall['mean_b']:.4f} (n={overall['n_b']}), "
                 f"diff={overall['mean_diff']:+.4f}, 95% CI [{overall['ci_lo']:+.4f}, {overall['ci_hi']:+.4f}] "
                 f"-> significant: {v}")
    lines.append("")
    lines.append("| dataset | prevalence (has_bare_cell) | n has-bare | n no-bare | "
                 "mean rouge5 (has) | mean rouge5 (no) | diff | 95% CI | significant |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for ds, (prevalence, r) in sorted(by_dataset.items()):
        if r["ci_lo"] is None:
            lines.append(f"| {ds} | {prevalence:.1%} | {r['n_a']} | {r['n_b']} | - | - | - | - | n/a (empty group) |")
            continue
        sig = "yes" if _is_significant(r) else "no"
        lines.append(f"| {ds} | {prevalence:.1%} | {r['n_a']} | {r['n_b']} | "
                     f"{r['mean_a']:.4f} | {r['mean_b']:.4f} | {r['mean_diff']:+.4f} | "
                     f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {sig} |")
    return "\n".join(lines)


def run_bare_cell_hypothesis(feature_rows, out_basename):
    """feature_rows: list of dicts with dataset/rouge5/has_bare_cell. Groups by dataset,
    runs unpaired_bootstrap has-bare-cell vs no-bare-cell within each dataset and overall."""
    by_ds = defaultdict(lambda: ([], []))  # ds -> (has_bare_scores, no_bare_scores)
    all_has, all_no = [], []
    prevalence = {}
    ds_counts = defaultdict(int)
    for row in feature_rows:
        ds = row["dataset"]
        ds_counts[ds] += 1
        score = row["rouge5"]
        if row["has_bare_cell"]:
            by_ds[ds][0].append(score)
            all_has.append(score)
        else:
            by_ds[ds][1].append(score)
            all_no.append(score)

    by_dataset_result = {}
    for ds in ds_counts:
        has_scores, no_scores = by_ds[ds]
        r = unpaired_bootstrap(has_scores, no_scores)
        prevalence[ds] = len(has_scores) / ds_counts[ds]
        by_dataset_result[ds] = (prevalence[ds], r)

    overall = unpaired_bootstrap(all_has, all_no)

    out = {
        "overall": overall,
        "by_dataset": {ds: {"prevalence_has_bare_cell": prevalence[ds], **r}
                       for ds, (_, r) in by_dataset_result.items()},
    }
    with open(out_basename + ".json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(out_basename + ".md", "w", encoding="utf-8") as f:
        f.write(format_bare_cell_markdown(by_dataset_result, overall))
    print(f"saved: {out_basename}.json + .md")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wceb", required=True, help="path to .../datasets/combined")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--oracle-csv", default="results/oracle/wceb.csv",
                    help="existing oracle run to join rouge5/n_blocks from (not recomputed here)")
    ap.add_argument("--pages", choices=["all", "worst"], default="all")
    ap.add_argument("--worst-threshold", type=float, default=0.5,
                    help="--pages worst selects oracle-csv rows with rouge5 below this")
    ap.add_argument("--threshold", type=float, default=0.5, help="overlap_labels token-overlap cutoff")
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--out", default="oracle_diagnosis", help="output basename for the features .csv")
    ap.add_argument("--stats-out", default=None,
                    help="if set, run the bare-cell hypothesis test and write here (.md/.json); "
                         "only meaningful with --pages all")
    ap.add_argument("--dump-diff", nargs="*", default=None,
                    help="page_ids to dump a pred-vs-gt token diff for (qualitative spot-check)")
    ap.add_argument("--diff-dir", default="results/oracle_diagnosis/diffs")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    oracle_rows = load_oracle_csv(args.oracle_csv)
    if args.pages == "worst":
        wanted = {key for key, row in oracle_rows.items() if float(row["rouge5"]) < args.worst_threshold}
        print(f"--pages worst: {len(wanted)} pages with rouge5 < {args.worst_threshold}")
    else:
        wanted = None  # all pages present in oracle_rows

    dump_ids = set(args.dump_diff) if args.dump_diff else set()
    if dump_ids:
        os.makedirs(args.diff_dir, exist_ok=True)

    csv_file = open(args.out + ".csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["dataset", "page_id", "rouge5", "n_blocks", "n_kept", "bare_cell_frac",
                     "has_bare_cell", "n_bare_total", "truncated_kept_frac"])

    feature_rows = []
    n_seen = 0
    for ds, page_id, html, gt in read_wceb(args.wceb, args.datasets):
        key = (ds, page_id)
        if key not in oracle_rows:
            continue
        if wanted is not None and key not in wanted:
            continue
        try:
            simpl, _mapping = parse_page(html)
            texts = block_texts(simpl)
            keep = overlap_labels(texts, gt, args.threshold, args.min_words)
            feats = page_features(simpl, keep)
        except Exception as e:
            print(f"[{ds}/{page_id[:8]}] skip: {e}")
            continue

        rouge5 = float(oracle_rows[key]["rouge5"])
        row = {"dataset": ds, "page_id": page_id, "rouge5": rouge5,
              "n_blocks": int(oracle_rows[key]["n_blocks"]), **feats}
        feature_rows.append(row)
        writer.writerow([ds, page_id, f"{rouge5:.6f}", row["n_blocks"], feats["n_kept"],
                         f"{feats['bare_cell_frac']:.6f}", int(feats["has_bare_cell"]),
                         feats["n_bare_total"], f"{feats['truncated_kept_frac']:.6f}"])
        n_seen += 1

        if page_id in dump_ids or f"{ds}/{page_id}" in dump_ids:
            from src.evaluation.blockReconstruction import reconstruct
            from src.evaluation.textMetrics import to_text
            body = reconstruct(simpl, _mapping, keep)
            pred = to_text(body)
            diff = difflib.unified_diff(tokenize(gt), tokenize(pred), lineterm="",
                                        fromfile="ground_truth", tofile="reconstructed")
            with open(os.path.join(args.diff_dir, f"{ds}_{page_id}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(diff))
            print(f"  dumped diff: {ds}/{page_id[:8]}")

        if n_seen % 500 == 0:
            print(f"  {n_seen} pages processed")

    csv_file.close()
    print(f"saved: {args.out}.csv  ({n_seen} pages)")

    if args.stats_out:
        run_bare_cell_hypothesis(feature_rows, args.stats_out)


if __name__ == "__main__":
    main()
