"""
Pulls every number used in analysis/*.md and analysis/figures/* directly from the source
JSON/CSV files under results_from_server/ and results/ (repo root), so nothing in the write-up
is hand-transcribed. Run once: `python analysis/collect_data.py`. Writes
analysis/consolidated_data.json, which generate_figures.py reads.

Deliberately reads files directly with json/csv, no dependency on src.evaluation modules (this
is a data-collection script, not part of the pipeline).
"""

import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RFS = os.path.join(ROOT, "results_from_server")

WINNING_RUNS = {
    "bilstm": "runs/sweep_bilstm/20260724-172733_bilstm_6b7c4d35",
    "gru": "runs/sweep_gru/20260728-150826_gru_c12a355b",
    "xlstm": "runs/sweep_xlstm/20260728-232824_xlstm_a44f24a6",
}


def load_json(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    data = {"archs": {}}

    summary = load_json(RFS, "results", "summary.json")
    by_arch = {row["arch"]: row for row in summary}

    sweep_leaderboards = {
        arch: load_json(RFS, "runs", f"sweep_{arch}", "summary.json")
        for arch in WINNING_RUNS
    }

    lodo_aggregates = {}
    for arch in WINNING_RUNS:
        path = os.path.join(RFS, "runs", f"lodo_{arch}", "aggregate.json")
        lodo_aggregates[arch] = load_json(path) if os.path.exists(path) else None

    winning_wceb = {
        arch: load_json(RFS, run_dir, "wceb.json")
        for arch, run_dir in WINNING_RUNS.items()
    }
    winning_metrics = {
        arch: load_json(RFS, run_dir, "metrics.json")
        for arch, run_dir in WINNING_RUNS.items()
    }
    winning_config = {
        arch: load_json(RFS, run_dir, "config.json")
        for arch, run_dir in WINNING_RUNS.items()
    }

    oracle = load_json(RFS, "results", "oracle.json")

    legacy = {
        "bilstm_combined3": load_json(RFS, "results_combined3", "wceb.json"),
        "bilstm_combined4": load_json(RFS, "results_combined4", "wceb.json"),
        "xlstm_legacy": load_json(RFS, "wceb_results.json"),
        "plain_minilm": load_json(RFS, "results", "wceb_minilm.json"),
    }

    cross_arch_compare = {}
    for pair in ["bilstm_vs_gru", "bilstm_vs_xlstm", "gru_vs_xlstm"]:
        path = os.path.join(RFS, "results", f"compare_{pair}.json")
        if os.path.exists(path):
            cross_arch_compare[pair] = load_json(path)

    oracle_vs_arch = {}
    for arch in WINNING_RUNS:
        path = os.path.join(ROOT, "analysis", "stats", f"oracle_vs_{arch}.json")
        if os.path.exists(path):
            oracle_vs_arch[arch] = load_json(path)

    # trainable_minilm: standalone experiment, no sweep/LODO, different path shapes -- own
    # top-level key rather than a WINNING_RUNS entry.
    trainable_minilm_run_dir = "runs/20260809-234527_trainable_minilm_bilstm_082e0309"
    trainable_minilm = {
        "config": load_json(RFS, trainable_minilm_run_dir, "config.json"),
        "metrics": load_json(RFS, trainable_minilm_run_dir, "metrics.json"),
        "wceb": load_json(RFS, "trainable_minilm", "wceb.json"),
        "oracle_vs": load_json(ROOT, "analysis", "stats", "oracle_vs_trainable_minilm.json"),
        "vs_bilstm_combined": load_json(ROOT, "analysis", "stats", "trainable_minilm_vs_bilstm.json"),
    }

    for arch in WINNING_RUNS:
        summary_row = dict(by_arch.get(arch) or {})
        # lodo_<arch>/aggregate.json is the fresher, authoritative Phase-B source -- summary.json
        # can lag behind it (seen in practice: GRU's summary.json Phase-B was a stale snapshot
        # from before this aggregate.json existed). Override phaseB_* fields from it when present.
        agg = lodo_aggregates[arch]
        if agg is not None:
            summary_row["phaseB_rouge5_mean"] = agg["rouge5"]["mean"]
            summary_row["phaseB_rouge5_std"] = agg["rouge5"]["std"]
            summary_row["phaseB_rouge_l_mean"] = agg["rouge_l"]["mean"]
            summary_row["phaseB_rouge_l_std"] = agg["rouge_l"]["std"]
            summary_row["phaseB_block_f1_mean"] = agg["block_f1"]["mean"]
            summary_row["phaseB_block_f1_std"] = agg["block_f1"]["std"]

        data["archs"][arch] = {
            "phaseA_phaseB_summary": summary_row,
            "sweep_leaderboard": sweep_leaderboards[arch],
            "lodo_aggregate": lodo_aggregates[arch],
            "winning_wceb": winning_wceb[arch],
            "winning_metrics": winning_metrics[arch],
            "winning_config": winning_config[arch],
        }

    data["oracle"] = oracle
    data["legacy"] = legacy
    data["cross_arch_compare"] = cross_arch_compare
    data["oracle_vs_arch"] = oracle_vs_arch
    data["trainable_minilm"] = trainable_minilm

    out_path = os.path.join(ROOT, "analysis", "consolidated_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
