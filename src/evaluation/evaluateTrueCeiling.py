"""
True-ceiling check for the oracle-ceiling investigation (see analysis/oracle_investigation.md):
how much ROUGE-5 headroom exists above the current silver-label oracle at the SAME block
segmentation/reconstruction granularity, independent of the labeling heuristic?

Starts from the silver-label oracle's own keep mask (overlap_labels) and runs a coordinate-
ascent greedy toggle search: repeatedly flip whichever single block's inclusion improves the
page's ROUGE-5 the most, until a full pass makes no more flips. This is a CONSERVATIVE LOWER
BOUND on the true ceiling -- it cannot find improvements that require two blocks to flip
jointly (e.g. a sentence split across adjacent bare table cells where neither cell alone
contains a 5-gram match). Do not read its output as "the" true ceiling, only as "at least
this much heuristic-driven headroom exists."

Intended to run on a small, targeted page sample (e.g. the oracle's worst-scoring tail), NOT
the full WCEB benchmark: every flip candidate re-does a full reconstruct()+to_text()+tokenize
pass (no incremental scoring -- out of scope for a bachelor-thesis-sized investigation), so
cost scales with n_blocks * n_passes per page. --max-blocks skips (and logs) pages above a
size where this becomes impractical, rather than silently truncating the search.

run: python -m src.evaluation.evaluateTrueCeiling --wceb src/evaluation/web-content-extraction-benchmark/datasets/combined \
        --page-ids results/oracle_diagnosis/worst_tail_features.csv --out results/trueCeiling/wceb_worst
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict

from src.evaluation.wcebLoader import read_wceb
from src.evaluation.textMetrics import rouge_n_f1, rouge_l_f1, to_text, tokenize
from src.evaluation.blockReconstruction import parse_page, block_texts, reconstruct, overlap_labels


def _rouge_n_pretokenized(pt, rt, n=5):
    """rouge_n_f1's core logic, taking already-tokenized pred/ref -- avoids re-tokenizing
    gt on every single flip candidate inside the search loop."""
    from collections import Counter
    if not pt and not rt:
        return 1.0
    ng = lambda t: Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
    pg, rg = ng(pt), ng(rt)
    if not pg or not rg:
        return 0.0
    overlap = sum((pg & rg).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / sum(pg.values()), overlap / sum(rg.values())
    return 2 * prec * rec / (prec + rec)


def greedy_toggle_search(simpl_blocks, mapping_blocks, keep_init, ref_toks, n=5,
                         max_passes=2, eps=1e-9):
    """Coordinate-ascent hill-climb on the binary keep mask, starting from keep_init (the
    silver-label oracle's own decision), locally maximizing rouge_n_f1 of the
    reconstruction against ref_toks (pre-tokenized ground truth). Sequential: flips made
    earlier in a pass are visible to later flips in the same pass. Returns
    (keep, best_score, n_evals, n_flips, converged)."""
    keep = list(keep_init)

    def score(k):
        body = reconstruct(simpl_blocks, mapping_blocks, k)
        pred_toks = tokenize(to_text(body))
        return _rouge_n_pretokenized(pred_toks, ref_toks, n)

    initial = best = score(keep)
    n_evals, n_flips = 1, 0
    converged = False
    for _ in range(max_passes):
        flips_this_pass = 0
        for i in range(len(keep)):
            keep[i] = 1 - keep[i]
            s = score(keep)
            n_evals += 1
            if s > best + eps:
                best = s
                flips_this_pass += 1
            else:
                keep[i] = 1 - keep[i]  # revert
        n_flips += flips_this_pass
        if flips_this_pass == 0:
            converged = True
            break
    return keep, initial, best, n_evals, n_flips, converged


def true_ceiling_page(html, gt, threshold, min_words, n, max_passes):
    simpl, mapping = parse_page(html)
    texts = block_texts(simpl)
    keep_init = overlap_labels(texts, gt, threshold, min_words)
    ref_toks = tokenize(gt)
    keep_final, oracle5, best5, n_evals, n_flips, converged = greedy_toggle_search(
        simpl, mapping, keep_init, ref_toks, n=n, max_passes=max_passes)
    body = reconstruct(simpl, mapping, keep_final)
    pred = to_text(body)
    bestL = rouge_l_f1(pred, gt)
    return keep_final, oracle5, best5, bestL, n_evals, n_flips, converged, len(simpl)


def load_page_ids(path):
    """--page-ids accepts either a wceb-style CSV with dataset/page_id columns (e.g.
    diagnoseOraclePages.py's worst-tail output, or the oracle wceb.csv itself) or a plain
    text file with one 'dataset,page_id' or 'dataset/page_id' per line."""
    wanted = set()
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        if "," in first and ("dataset" in first or "page_id" in first):
            for row in csv.DictReader(f):
                wanted.add((row["dataset"], row["page_id"]))
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sep = "," if "," in line else "/"
                ds, pid = line.split(sep, 1)
                wanted.add((ds, pid))
    return wanted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wceb", required=True, help="path to .../datasets/combined")
    ap.add_argument("--page-ids", required=True,
                    help="CSV (dataset,page_id columns) or text file restricting the search "
                         "to a specific page sample -- this script is not meant to run on the "
                         "full benchmark, see module docstring")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-passes", type=int, default=2)
    ap.add_argument("--max-blocks", type=int, default=2500,
                    help="pages with more blocks than this are skipped (and logged), not "
                         "silently truncated -- full search cost scales with n_blocks")
    ap.add_argument("--out", default="results/trueCeiling/wceb_worst")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    wanted = load_page_ids(args.page_ids)
    datasets = sorted({ds for ds, _ in wanted})
    print(f"true-ceiling search over {len(wanted)} pages across {len(datasets)} datasets, "
         f"max_passes={args.max_passes}, max_blocks={args.max_blocks}")

    csv_file = open(args.out + ".csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["dataset", "page_id", "n_blocks", "rouge5_oracle", "rouge5_true_ceiling",
                     "rouge_l_true_ceiling", "n_evals", "n_flips", "converged",
                     "skipped_too_large", "wall_time_sec"])

    by_ds_gain = defaultdict(list)
    n_done = n_skipped = 0

    for ds, page_id, html, gt in read_wceb(args.wceb, datasets):
        if (ds, page_id) not in wanted:
            continue

        simpl, _ = parse_page(html)
        if len(simpl) > args.max_blocks:
            print(f"[{ds}/{page_id[:8]}] skip: n_blocks={len(simpl)} > --max-blocks {args.max_blocks}")
            writer.writerow([ds, page_id, len(simpl), "", "", "", "", "", "", 1, ""])
            n_skipped += 1
            continue

        t0 = time.time()
        try:
            _keep, oracle_r5, r5, rL, n_evals, n_flips, converged, n_blocks = true_ceiling_page(
                html, gt, args.threshold, args.min_words, args.n, args.max_passes)
        except Exception as e:
            print(f"[{ds}/{page_id[:8]}] error: {e}")
            continue
        wall = time.time() - t0

        writer.writerow([ds, page_id, n_blocks, f"{oracle_r5:.6f}", f"{r5:.6f}", f"{rL:.6f}",
                         n_evals, n_flips, int(converged), 0, f"{wall:.2f}"])
        csv_file.flush()
        by_ds_gain[ds].append(r5 - oracle_r5)
        n_done += 1
        print(f"[{ds}/{page_id[:8]}] n_blocks={n_blocks} oracle={oracle_r5:.4f} "
             f"true_ceiling={r5:.4f} (+{r5 - oracle_r5:.4f}) flips={n_flips} "
             f"converged={converged} wall={wall:.1f}s")

    csv_file.close()

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    all_gains = [g for gs in by_ds_gain.values() for g in gs]
    summary = {
        "n_pages": n_done,
        "n_skipped_too_large": n_skipped,
        "max_passes": args.max_passes,
        "note": "conservative lower bound -- single-flip coordinate ascent, see module docstring",
        "mean_gain_over_oracle": round(mean(all_gains), 6),
        "by_dataset_mean_gain": {ds: round(mean(gs), 6) for ds, gs in sorted(by_ds_gain.items())},
    }
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== true ceiling vs. silver-label oracle, n={n_done} pages "
         f"(+{n_skipped} skipped, too large) ===")
    print(f"mean gain: {summary['mean_gain_over_oracle']:+.4f}")
    for ds, g in summary["by_dataset_mean_gain"].items():
        print(f"  {ds:20s}  {g:+.4f}")
    print(f"\nsaved: {args.out}.csv  +  {args.out}.json")


if __name__ == "__main__":
    main()
