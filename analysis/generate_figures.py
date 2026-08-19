"""
Generates every figure referenced by analysis/*.md from analysis/consolidated_data.json (built
by collect_data.py). Reads only local files, no model/cluster dependency -- rerun any time after
collect_data.py to regenerate everything, e.g. once the GRU LODO rerun lands.

Run: python analysis/collect_data.py && python analysis/generate_figures.py
"""

import json
import math
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "analysis", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --- palette (dataviz skill reference instance, light mode, fixed categorical order) ---
COLOR = {
    "bilstm": "#2a78d6",   # slot 1: blue
    "gru": "#1baf7a",      # slot 2: aqua
    "xlstm": "#eda100",    # slot 3: yellow
    "trainable_minilm": "#008300",  # slot 4: green
    "oracle": "#898781",   # muted ink, not a categorical slot -- it's a reference line, not a series
    "legacy": "#c3c2b7",   # baseline/axis muted tone, for "old/legacy" bars
}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9"}
SURFACE = "#fcfcfb"
ARCH_LABEL = {"bilstm": "BiLSTM", "gru": "GRU", "xlstm": "xLSTM", "trainable_minilm": "Trainable-MiniLM"}
DATASETS = ["cetd", "cleaneval", "cleanportaleval", "dragnet", "google-trends-2017",
            "l3s-gn1", "readability", "scrapinghub"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": INK["grid"],
    "axes.labelcolor": INK["secondary"],
    "text.color": INK["primary"],
    "xtick.color": INK["secondary"],
    "ytick.color": INK["secondary"],
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.grid": True,
    "grid.color": INK["grid"],
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "font.size": 10,
})


def style_axes(ax, ygrid_only=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    if ygrid_only:
        ax.xaxis.grid(False)
    ax.tick_params(length=0)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png / {name}.pdf")


def load():
    with open(os.path.join(ROOT, "analysis", "consolidated_data.json"), encoding="utf-8") as f:
        return json.load(f)


ARCHES = ["bilstm", "gru", "xlstm"]


def fig_phaseA_phaseB_oracle(d):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    x = range(len(ARCHES))
    width = 0.32
    a_vals = [d["archs"][a]["phaseA_phaseB_summary"]["phaseA_rouge5"] for a in ARCHES]
    b_vals = [d["archs"][a]["phaseA_phaseB_summary"]["phaseB_rouge5_mean"] for a in ARCHES]
    b_err = [d["archs"][a]["phaseA_phaseB_summary"]["phaseB_rouge5_std"] for a in ARCHES]

    ax.bar([i - width / 2 for i in x], a_vals, width, label="Phase A (WebMainBench-trained)",
           color=[COLOR[a] for a in ARCHES], edgecolor="none")
    ax.bar([i + width / 2 for i in x], b_vals, width, yerr=b_err, capsize=3,
           label="Phase B (LODO, mean ± std over 8 folds)",
           color=[COLOR[a] for a in ARCHES], edgecolor=INK["primary"], linewidth=0.8, alpha=0.55,
           error_kw={"ecolor": INK["secondary"], "linewidth": 1})

    oracle_f1 = d["oracle"]["rouge5"]["overall_f1"]
    ax.axhline(oracle_f1, color=INK["muted"], linestyle="--", linewidth=1.2, zorder=0,
               label=f"oracle ceiling ({oracle_f1:.3f})")

    for i, (a, b) in enumerate(zip(a_vals, b_vals)):
        ax.text(i - width / 2, a + 0.02, f"{a:.3f}", ha="center", fontsize=8.5, color=INK["primary"])
        ax.text(i + width / 2, b + b_err[i] + 0.02, f"{b:.3f}",
                ha="center", fontsize=8.5, color=INK["secondary"])

    ax.set_xticks(list(x))
    ax.set_xticklabels([ARCH_LABEL[a] for a in ARCHES])
    ax.set_ylabel("ROUGE-5 F1 (full WCEB, 3985 pages)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase A vs. Phase B vs. oracle ceiling")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, loc="upper center", fontsize=8.5,
              bbox_to_anchor=(0.5, -0.12), ncol=1)
    style_axes(ax)
    save(fig, "phaseA_vs_phaseB_vs_oracle")


def fig_per_dataset_rouge5(d):
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(DATASETS)
    width = 0.25
    x = range(n)
    for i, a in enumerate(ARCHES):
        by_ds = d["archs"][a]["winning_wceb"]["rouge5"]["by_dataset"]
        vals = [by_ds[ds]["f1"] for ds in DATASETS]
        ax.bar([xi + (i - 1) * width for xi in x], vals, width, label=ARCH_LABEL[a],
               color=COLOR[a], edgecolor="none")

    oracle_by_ds = d["oracle"]["rouge5"]["by_dataset"]
    oracle_vals = [oracle_by_ds[ds]["f1"] for ds in DATASETS]
    ax.scatter(list(x), oracle_vals, marker="_", s=400, color=INK["primary"], linewidths=1.6,
               zorder=5, label="oracle ceiling")

    ax.set_xticks(list(x))
    ax.set_xticklabels(DATASETS, rotation=30, ha="right")
    ax.set_ylabel("ROUGE-5 F1 (Phase A)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-dataset ROUGE-5, Phase A, vs. oracle ceiling")
    ax.legend(frameon=False, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.32))
    style_axes(ax)
    save(fig, "per_dataset_rouge5")


def fig_lodo_per_fold(d):
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(DATASETS)
    width = 0.26
    x = range(n)
    for i, a in enumerate(ARCHES):
        per_fold = {r["fold"]: r["rouge5"] for r in d["archs"][a]["lodo_aggregate"]["per_fold"]}
        vals = [per_fold[ds] for ds in DATASETS]
        ax.bar([xi + (i - 1) * width for xi in x], vals, width, label=ARCH_LABEL[a],
               color=COLOR[a], edgecolor="none")

    ax.set_xticks(list(x))
    ax.set_xticklabels(DATASETS, rotation=30, ha="right")
    ax.set_ylabel("ROUGE-5 F1 (held-out fold)")
    ax.set_ylim(0, 1.05)
    ax.set_title("LODO per-fold ROUGE-5")
    ax.legend(frameon=False, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.32))
    style_axes(ax)
    save(fig, "lodo_per_fold")


def fig_quality_vs_cost(d):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    all_arches = ARCHES + ["trainable_minilm"]
    params = [d["archs"][a]["phaseA_phaseB_summary"]["phaseA_n_params"] for a in ARCHES]
    params.append(d["trainable_minilm"]["wceb"]["n_params"])
    pps = [d["archs"][a]["phaseA_phaseB_summary"]["phaseA_pages_per_sec"] for a in ARCHES]
    pps.append(d["trainable_minilm"]["wceb"]["throughput"]["pages_per_sec"])
    rouge5 = [d["archs"][a]["phaseA_phaseB_summary"]["phaseA_rouge5"] for a in ARCHES]
    rouge5.append(d["trainable_minilm"]["wceb"]["rouge5"]["overall_f1"])

    # log-scaled marker area: trainable-MiniLM's 118.2M params vs. the other three's <700K
    # would make linear scaling either invisible or comically enormous.
    log_p = [math.log10(p) for p in params]
    max_lp, min_lp = max(log_p), min(log_p)
    sizes = [120 + 900 * (lp - min_lp) / (max_lp - min_lp) for lp in log_p]

    # xLSTM and GRU sit close together on the x-axis -- offset xLSTM's label below its marker
    # instead of above, so the two annotations don't overlap.
    label_offset = {"xlstm": (0, -34)}
    for a, x, y, s, p in zip(all_arches, pps, rouge5, sizes, params):
        ax.scatter(x, y, s=s, color=COLOR[a], edgecolor=INK["primary"], linewidth=0.8, zorder=3)
        ax.annotate(f"{ARCH_LABEL[a]}\n{p:,} params",
                    (x, y), textcoords="offset points", xytext=label_offset.get(a, (0, 22)),
                    ha="center", fontsize=8.5, color=INK["primary"])

    ax.set_xlabel("Inference throughput (pages/sec, higher = cheaper)")
    ax.set_ylabel("ROUGE-5 F1 (Phase A)")
    ax.set_title("Quality vs. cost: marker size = log-scaled parameter count")
    ax.set_xlim(min(pps) - 0.35, max(pps) + 0.35)
    ax.set_ylim(0.60, 0.99)
    style_axes(ax, ygrid_only=False)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    save(fig, "quality_vs_cost")


def fig_training_cost(d):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    wall = [d["archs"][a]["winning_metrics"]["wall_time_sec"] for a in ARCHES]
    params = [d["archs"][a]["phaseA_phaseB_summary"]["phaseA_n_params"] for a in ARCHES]

    ax1.bar(range(len(ARCHES)), wall, color=[COLOR[a] for a in ARCHES], width=0.55)
    for i, w in enumerate(wall):
        ax1.text(i, w + max(wall) * 0.02, f"{w:,.0f}s", ha="center", fontsize=8.5, color=INK["primary"])
    ax1.set_xticks(range(len(ARCHES)))
    ax1.set_xticklabels([ARCH_LABEL[a] for a in ARCHES])
    ax1.set_ylabel("Training wall-clock, winning config (s)")
    ax1.set_title("Training cost")
    style_axes(ax1)

    ax2.bar(range(len(ARCHES)), params, color=[COLOR[a] for a in ARCHES], width=0.55)
    for i, p in enumerate(params):
        ax2.text(i, p + max(params) * 0.02, f"{p:,}", ha="center", fontsize=8.5, color=INK["primary"])
    ax2.set_xticks(range(len(ARCHES)))
    ax2.set_xticklabels([ARCH_LABEL[a] for a in ARCHES])
    ax2.set_ylabel("Parameters")
    ax2.set_title("Model size")
    style_axes(ax2)

    fig.tight_layout()
    save(fig, "training_cost")


def fig_sweep_sensitivity(d):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for ax, a in zip(axes, ARCHES):
        rows = [r for r in d["archs"][a]["sweep_leaderboard"] if "wceb_rouge5" in r]
        val_f1 = [r["val_f1"] for r in rows]
        wceb = [r["wceb_rouge5"] for r in rows]
        run_dirs = [r["run_dir"] for r in rows]
        winner = d["archs"][a]["phaseA_phaseB_summary"]["phaseA_run_dir"]
        colors = [INK["primary"] if rd == winner else COLOR[a] for rd in run_dirs]
        sizes = [70 if rd == winner else 45 for rd in run_dirs]
        ax.scatter(val_f1, wceb, color=colors, s=sizes, zorder=3,
                   edgecolor=INK["primary"], linewidth=0.5)
        ax.set_title(ARCH_LABEL[a])
        ax.set_xlabel("val_f1 (cheap proxy)")
        style_axes(ax, ygrid_only=False)
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
    axes[0].set_ylabel("wceb_rouge5 (expensive, top-K only)")
    fig.suptitle("Sweep top-K: cheap proxy metric vs. real WCEB score  (black = eventual winner)",
                 fontsize=10.5, color=INK["primary"])
    fig.tight_layout()
    save(fig, "sweep_sensitivity")


def fig_training_curves(d):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a in ARCHES:
        hist = d["archs"][a]["winning_metrics"]["history"]
        epochs = [h["epoch"] for h in hist]
        val_f1 = [h["val_f1"] for h in hist]
        ax.plot(epochs, val_f1, color=COLOR[a], linewidth=2, marker="o", markersize=4,
                label=ARCH_LABEL[a])
        best_epoch = d["archs"][a]["winning_metrics"]["best_epoch"]
        best_f1 = hist[best_epoch]["val_f1"]
        ax.scatter([best_epoch], [best_f1], s=90, facecolor="none", edgecolor=COLOR[a],
                   linewidth=1.6, zorder=4)

    ax.set_xlabel("epoch")
    ax.set_ylabel("validation block-F1")
    ax.set_title("Training curves, winning config per architecture\n(open circle = checkpointed best epoch)")
    ax.legend(frameon=False, fontsize=8.5)
    style_axes(ax)
    save(fig, "training_curves")


def fig_legacy_vs_current_bilstm(d):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = ["combined3\n(2026-06-08)", "combined4\n(2026-06-08)", "Phase A\n(sweep-tuned)"]
    vals = [
        d["legacy"]["bilstm_combined3"]["rouge"]["overall_f1"],
        d["legacy"]["bilstm_combined4"]["rouge"]["overall_f1"],
        d["archs"]["bilstm"]["phaseA_phaseB_summary"]["phaseA_rouge5"],
    ]
    colors = [COLOR["legacy"], COLOR["legacy"], COLOR["bilstm"]]
    ax.bar(range(3), vals, color=colors, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, color=INK["primary"])
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Overall ROUGE-5 F1 (full WCEB)")
    ax.set_ylim(0, 0.85)
    ax.set_title("BiLSTM: legacy pre-sweep runs vs. sweep-tuned Phase A")
    style_axes(ax)
    save(fig, "legacy_vs_current_bilstm")


def fig_legacy_vs_current_xlstm(d):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    labels = ["legacy\n(2026-06-11, keep_th=0.3)", "Phase A\n(sweep-tuned, keep_th=0.5)"]
    vals = [
        d["legacy"]["xlstm_legacy"]["rouge"]["overall_f1"],
        d["archs"]["xlstm"]["phaseA_phaseB_summary"]["phaseA_rouge5"],
    ]
    colors = [COLOR["legacy"], COLOR["xlstm"]]
    ax.bar(range(2), vals, color=colors, width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, color=INK["primary"])
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Overall ROUGE-5 F1 (full WCEB)")
    ax.set_ylim(0, 1.0)
    ax.set_title("xLSTM: legacy pre-sweep run vs. sweep-tuned Phase A")
    style_axes(ax)
    save(fig, "legacy_vs_current_xlstm")


def fig_frozen_vs_finetuned_minilm(d):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    labels = ["frozen MiniLM\n(2026-06-06)", "fine-tuned MiniLM\n(this run)"]
    vals = [
        d["legacy"]["plain_minilm"]["overall_f1"],
        d["trainable_minilm"]["wceb"]["rouge5"]["overall_f1"],
    ]
    colors = [COLOR["legacy"], COLOR["trainable_minilm"]]
    ax.bar(range(2), vals, color=colors, width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, color=INK["primary"])
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Overall ROUGE-5 F1 (full WCEB)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Same tagger, frozen vs. fine-tuned MiniLM embedder")
    style_axes(ax)
    save(fig, "frozen_vs_finetuned_minilm")


def fig_finetuning_cost(d):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    all_arches = ARCHES + ["trainable_minilm"]
    wall = [d["archs"][a]["winning_metrics"]["wall_time_sec"] for a in ARCHES]
    wall.append(d["trainable_minilm"]["metrics"]["wall_time_sec"])
    params = [d["archs"][a]["phaseA_phaseB_summary"]["phaseA_n_params"] for a in ARCHES]
    params.append(d["trainable_minilm"]["wceb"]["n_params"])
    colors = [COLOR[a] for a in all_arches]
    labels = [ARCH_LABEL[a] for a in all_arches]

    ax1.bar(range(len(all_arches)), wall, color=colors, width=0.55)
    ax1.set_yscale("log")
    for i, w in enumerate(wall):
        ax1.text(i, w * 1.15, f"{w:,.0f}s", ha="center", fontsize=8.5, color=INK["primary"])
    ax1.set_xticks(range(len(all_arches)))
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Training wall-clock, winning config (s, log scale)")
    ax1.set_title("Training cost")
    style_axes(ax1)

    ax2.bar(range(len(all_arches)), params, color=colors, width=0.55)
    ax2.set_yscale("log")
    for i, p in enumerate(params):
        ax2.text(i, p * 1.15, f"{p:,}", ha="center", fontsize=8.5, color=INK["primary"])
    ax2.set_xticks(range(len(all_arches)))
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("Parameters (log scale)")
    ax2.set_title("Model size")
    style_axes(ax2)

    fig.suptitle("Cost of fine-tuning the embedder vs. the three frozen-embedding winners")
    fig.tight_layout()
    save(fig, "finetuning_cost")


def main():
    d = load()
    fig_phaseA_phaseB_oracle(d)
    fig_per_dataset_rouge5(d)
    fig_lodo_per_fold(d)
    fig_quality_vs_cost(d)
    fig_training_cost(d)
    fig_sweep_sensitivity(d)
    fig_training_curves(d)
    fig_legacy_vs_current_bilstm(d)
    fig_legacy_vs_current_xlstm(d)
    fig_frozen_vs_finetuned_minilm(d)
    fig_finetuning_cost(d)


if __name__ == "__main__":
    main()
