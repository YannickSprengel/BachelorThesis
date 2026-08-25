"""
Sweep the keep-decision threshold against a single trained model without redoing the
expensive simplify/embed/forward pass per threshold value -- evaluate*.py's run_eval()
bakes the threshold into predict_page() itself, so testing N thresholds the naive way
(N separate `evaluate*.py --threshold ...` invocations) means N full WCEB passes for
something that only differs in how the same per-block sigmoid score gets binarized.

Here the simplify->embed->forward pass runs once per page; every threshold in --thresholds
then only re-does keep-mask/reconstruct/score, which is nearly free by comparison.

run:  python -m src.evaluation.sweepThreshold --arch bilstm \
        --model runs/sweep_bilstm_dom/<run>/model.pt --config runs/sweep_bilstm_dom/<run>/config.json \
        --wceb src/evaluation/web-content-extraction-benchmark/datasets/combined \
        --thresholds 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5 --out results/bilstm_dom_threshold_sweep
"""

import argparse
import json
import statistics
import time
from collections import defaultdict

import torch

from src.evaluation.wcebLoader import read_wceb
from src.evaluation.textMetrics import rouge_n_f1, rouge_l_f1, to_text
from src.evaluation.blockReconstruction import parse_page, block_texts, reconstruct, overlap_labels, prf
from src.data.combinedLMEmbedder import embed_blocks
from src.models.lstm.biLSTMWithMiniLM import BiLSTMTagger
from src.models.lstm.trainLSTM import build_model as build_bilstm
from src.models.gru.biGRUWithMiniLM import BiGRUTagger
from src.models.gru.trainGRU import build_model as build_gru
from src.models.xlstm.xLSTMWithMiniLM import XLSTMTagger
from src.models.xlstm.trainxLSTM import build_model as build_xlstm

ARCH_BUILDERS = {"bilstm": build_bilstm, "gru": build_gru, "xlstm": build_xlstm}
ARCH_DEFAULTS = {"bilstm": BiLSTMTagger, "gru": BiGRUTagger, "xlstm": XLSTMTagger}


@torch.no_grad()
def page_probs(model, html, device, embed_fn=embed_blocks):
    """One simplify+embed+forward pass -> (simpl, mapping, texts, probs). The expensive part."""
    simpl, mapping = parse_page(html)
    emb = torch.as_tensor(embed_fn(simpl), dtype=torch.float32, device=device).unsqueeze(0)
    probs = torch.sigmoid(model(emb)).squeeze(0).cpu()
    return simpl, mapping, block_texts(simpl), probs


def score_at_threshold(simpl, mapping, texts, probs, gt, threshold, label_threshold, n):
    keep = probs > threshold
    body = reconstruct(simpl, mapping, keep)
    pred_text = to_text(body)
    s5 = rouge_n_f1(pred_text, gt, n=n)
    sL = rouge_l_f1(pred_text, gt)
    gt_lab = overlap_labels(texts, gt, label_threshold)
    pred_lab = [int(k) for k in keep]
    tp = sum(p == 1 and g == 1 for p, g in zip(pred_lab, gt_lab))
    fp = sum(p == 1 and g == 0 for p, g in zip(pred_lab, gt_lab))
    fn = sum(p == 0 and g == 1 for p, g in zip(pred_lab, gt_lab))
    return s5, sL, tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(ARCH_BUILDERS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=None, help="run dir's config.json; omit only for a "
                    "default-hyperparameter checkpoint")
    ap.add_argument("--wceb", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5",
                    help="comma-separated sigmoid cutoffs to evaluate in one pass")
    ap.add_argument("--label-threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="output basename for a summary .json")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="force a device instead of auto-detecting; use cpu if the node's GPU "
                         "hits a cuDNN/SM compatibility error (CUDA_VISIBLE_DEVICES doesn't "
                         "reliably prevent this under SLURM, this flag does)")
    args = ap.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.config:
        with open(args.config) as f:
            model = ARCH_BUILDERS[args.arch](json.load(f)).to(device)
    else:
        model = ARCH_DEFAULTS[args.arch]().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"device={device}  arch={args.arch}  thresholds={thresholds}")

    per_t = {t: {"s5": [], "sL": [], "tp": 0, "fp": 0, "fn": 0} for t in thresholds}
    n_pages = 0
    t_extract = 0.0

    for ds, page_id, html, gt in read_wceb(args.wceb, args.datasets):
        try:
            t0 = time.perf_counter()
            simpl, mapping, texts, probs = page_probs(model, html, device)
            t_extract += time.perf_counter() - t0
        except Exception as e:
            print(f"[{ds}/{page_id[:8]}] skip: {e}")
            continue

        n_pages += 1
        for t in thresholds:
            s5, sL, tp, fp, fn = score_at_threshold(simpl, mapping, texts, probs, gt, t,
                                                      args.label_threshold, args.n)
            d = per_t[t]
            d["s5"].append(s5); d["sL"].append(sL)
            d["tp"] += tp; d["fp"] += fp; d["fn"] += fn

        if n_pages % 200 == 0:
            print(f"  {n_pages} pages  extract_sec/page={t_extract / n_pages:.4f}", flush=True)

    print(f"\n=== threshold sweep, n_pages={n_pages}, extraction ran once/page "
          f"({t_extract:.1f}s total, {t_extract / max(n_pages, 1):.4f}s/page) ===")
    print(f"{'threshold':>10} {'rouge5':>8} {'rougeL':>8} {'blockP':>8} {'blockR':>8} {'blockF1':>8}")
    results = []
    for t in thresholds:
        d = per_t[t]
        mean5 = sum(d["s5"]) / len(d["s5"]) if d["s5"] else 0.0
        meanL = sum(d["sL"]) / len(d["sL"]) if d["sL"] else 0.0
        bP, bR, bF = prf(d["tp"], d["fp"], d["fn"])
        print(f"{t:>10.3f} {mean5:>8.4f} {meanL:>8.4f} {bP:>8.4f} {bR:>8.4f} {bF:>8.4f}")
        results.append({"threshold": t, "rouge5": round(mean5, 4), "rouge_l": round(meanL, 4),
                        "block_precision": round(bP, 4), "block_recall": round(bR, 4),
                        "block_f1": round(bF, 4), "tp": d["tp"], "fp": d["fp"], "fn": d["fn"]})

    if args.out:
        with open(args.out + ".json", "w", encoding="utf-8") as f:
            json.dump({"arch": args.arch, "model": args.model, "n_pages": n_pages,
                       "results": results}, f, indent=2)
        print(f"\nsaved: {args.out}.json")


if __name__ == "__main__":
    main()
