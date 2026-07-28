"""
run: python -m src.evaluation.aggregateResults --runs-dir runs --oracle results/oracle/wceb.json \
        --out results/summary

Scans:
  - --oracle: oracle ceiling JSON (evaluateOracleCeiling.py output) -- skipped with a warning if
    the path doesn't exist yet (it may not, e.g. if the oracle run is still in progress).
  - <runs-dir>/sweep_<arch>/summary.csv -- Phase A per-arch sweep leaderboard (sweep.py). Only
    rows with a non-empty wceb_rouge5 got a full WCEB eval (sweep.py's top-K); the highest of
    those is this architecture's reported Phase-A result.
  - <runs-dir>/lodo_<arch>/aggregate.json -- Phase B per-arch leave-one-dataset-out aggregate
    (aggregateLODO.py).

Produces one consolidated table -- results/summary.{csv,json,md} -- one row per architecture:
oracle ceiling reference, best Phase-A WCEB rouge5/rouge_l/block_f1/pages_per_sec/n_params, and
Phase-B LODO mean+-std rouge5/rouge_l/block_f1. This is the thesis's core results table and the
source data for a quality-vs-cost view (rouge5 vs n_params, rouge5 vs pages_per_sec).

Deliberately does NOT glob over results*/ -- the legacy results_combined*/ directories at the
repo root use an older, incompatible schema (single `rouge` column, no `arch` field) and would
silently corrupt this table if swept in. Inputs here are explicit, narrow paths only.
"""

import argparse
import csv
import glob
import json
import os


def load_oracle(path):
    if not path or not os.path.exists(path):
        print(f"[aggregateResults] oracle ceiling not found at {path!r}, skipping "
              f"(run evaluateOracleCeiling.py first if you want it in the table)")
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {"rouge5": data["rouge5"]["overall_f1"], "rouge_l": data["rouge_l"]["overall_f1"]}


def best_phase_a_row(summary_csv_path):
    """Highest wceb_rouge5 among rows that actually got a full WCEB eval (sweep.py's top-K)."""
    with open(summary_csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    evaluated = [r for r in rows if r.get("wceb_rouge5")]
    if not evaluated:
        return None
    return max(evaluated, key=lambda r: float(r["wceb_rouge5"]))


def discover_archs(runs_dir):
    sweep_archs = {os.path.basename(p)[len("sweep_"):]
                   for p in glob.glob(os.path.join(runs_dir, "sweep_*")) if os.path.isdir(p)}
    lodo_archs = {os.path.basename(p)[len("lodo_"):]
                  for p in glob.glob(os.path.join(runs_dir, "lodo_*")) if os.path.isdir(p)}
    return sorted(sweep_archs | lodo_archs)


def build_rows(runs_dir, oracle):
    rows = []
    for arch in discover_archs(runs_dir):
        row = {"arch": arch}
        if oracle:
            row["oracle_rouge5"] = oracle["rouge5"]
            row["oracle_rouge_l"] = oracle["rouge_l"]

        sweep_summary = os.path.join(runs_dir, f"sweep_{arch}", "summary.csv")
        if os.path.exists(sweep_summary):
            best = best_phase_a_row(sweep_summary)
            if best:
                row["phaseA_rouge5"] = float(best["wceb_rouge5"])
                row["phaseA_rouge_l"] = float(best["wceb_rouge_l"])
                row["phaseA_block_f1"] = float(best["wceb_block_f1"])
                row["phaseA_pages_per_sec"] = float(best["wceb_pages_per_sec"])
                row["phaseA_n_params"] = int(float(best.get("wceb_n_params") or 0)) or None
                row["phaseA_run_dir"] = best["run_dir"]
            else:
                print(f"[aggregateResults] {sweep_summary}: no row has a WCEB eval yet, "
                      f"skipping Phase A for '{arch}'")
        else:
            print(f"[aggregateResults] no sweep summary for '{arch}' at {sweep_summary}")

        lodo_agg = os.path.join(runs_dir, f"lodo_{arch}", "aggregate.json")
        if os.path.exists(lodo_agg):
            with open(lodo_agg, encoding="utf-8") as f:
                agg = json.load(f)
            row["phaseB_rouge5_mean"] = agg["rouge5"]["mean"]
            row["phaseB_rouge5_std"] = agg["rouge5"]["std"]
            row["phaseB_rouge_l_mean"] = agg["rouge_l"]["mean"]
            row["phaseB_rouge_l_std"] = agg["rouge_l"]["std"]
            row["phaseB_block_f1_mean"] = agg["block_f1"]["mean"]
            row["phaseB_block_f1_std"] = agg["block_f1"]["std"]
            if agg.get("per_fold"):
                row["phaseB_n_params"] = agg["per_fold"][0].get("n_params")
        else:
            print(f"[aggregateResults] no LODO aggregate for '{arch}' at {lodo_agg}")

        rows.append(row)
    return rows


def format_markdown(rows):
    lines = ["# Architecture comparison — quality vs. cost", ""]
    lines.append("Oracle ceiling: the best any tagger could score with perfect keep/drop decisions "
                  "(see evaluateOracleCeiling.py) -- a trained model can only approach this, never "
                  "beat it. Phase A: best hyperparameter-sweep config, trained on WebMainBench only, "
                  "evaluated on the full WCEB benchmark. Phase B: the same config retrained with "
                  "leave-one-dataset-out (LODO) cross-validation using WCEB's own source datasets as "
                  "extra training data -- mean +/- std across the 8 held-out folds.")
    lines.append("")
    lines.append("| arch | oracle rouge5 | Phase A rouge5 | Phase B rouge5 (mean+/-std) | "
                 "params | pages/sec |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        oracle = f"{r['oracle_rouge5']:.4f}" if "oracle_rouge5" in r else "—"
        a = f"{r['phaseA_rouge5']:.4f}" if "phaseA_rouge5" in r else "—"
        if "phaseB_rouge5_mean" in r:
            b = f"{r['phaseB_rouge5_mean']:.4f} +/- {r['phaseB_rouge5_std']:.4f}"
        else:
            b = "—"
        params = f"{r['phaseA_n_params']:,}" if r.get("phaseA_n_params") else "—"
        pps = f"{r['phaseA_pages_per_sec']:.2f}" if "phaseA_pages_per_sec" in r else "—"
        lines.append(f"| {r['arch']} | {oracle} | {a} | {b} | {params} | {pps} |")
    lines.append("")
    lines.append("Full per-metric detail (ROUGE-L, block-level P/R/F1, run dirs for provenance) is in "
                 "the accompanying summary.json -- this table is the at-a-glance version.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--oracle", default="results/oracle/wceb.json")
    ap.add_argument("--out", default="results/summary")
    args = ap.parse_args()

    oracle = load_oracle(args.oracle)
    rows = build_rows(args.runs_dir, oracle)

    if not rows:
        print(f"[aggregateResults] no sweep_*/lodo_* directories found under {args.runs_dir!r}, "
              f"nothing to aggregate yet")
        return

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    all_keys = sorted({k for row in rows for k in row.keys()})
    with open(args.out + ".csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    md = format_markdown(rows)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write(md)

    print(md)
    print(f"\nsaved: {args.out}.csv  +  {args.out}.json  +  {args.out}.md")


if __name__ == "__main__":
    main()
