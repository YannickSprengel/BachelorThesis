"""
Benchmark the trained model on WCEB (Bevendorff et al., 2023), reporting three things:

  1. ROUGE-5 F1 and ROUGE-L F1 (jieba)
  2. Block-level Precision / Recall / F1 of the model's keep/drop decisions   [model only]
  3. Extraction throughput: pages/sec + mean/median sec per page, with hardware/device.
        Only the extraction (simplify + embed + predict + reconstruct) is timed
        file I/O, text conversion and metric computation are excluded.

    How to run: python -m src.evaluation.evaluateXLSTM --model model.pt \
        --wceb src/evaluation/wceb_data/combined --out results/wceb

See src.evaluation.evaluateOracleCeiling for a model-free version of this that
reconstructs from gold labels instead of model predictions, to check how much of the
score ceiling comes from preprocessing/reconstruction rather than the tagger.
"""

import argparse

import torch

from src.evaluation.evalCommon import run_eval, print_report
from src.models.xlstm.xLSTMWithMiniLM import XLSTMTagger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--wceb", required=True, help="path to .../datasets/combined")
    ap.add_argument("--datasets", nargs="*", default=None, help="subset names (default: all)")
    ap.add_argument("--n", type=int, default=5)              # ROUGE-N, paper uses N=5
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="model keep-decision threshold (sigmoid cutoff); this is the one to tune")
    ap.add_argument("--label-threshold", type=float, default=0.5,
                    help="FIXED overlap cutoff for the silver labels; keep constant when tuning --threshold")
    ap.add_argument("--out", default="wceb_results", help="output basename for .csv and .json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = XLSTMTagger().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  xLSTM params={n_params:,}")
    print(f"keep-threshold={args.threshold}  label-threshold(silver)={args.label_threshold}")
    print(f"(ROUGE-{args.n} F1 + ROUGE-L F1, jieba  +  block-level P/R/F1  +  throughput)")

    summary = run_eval(model, args.wceb, datasets=args.datasets, n=args.n, threshold=args.threshold,
                        label_threshold=args.label_threshold, device=device, model_path=args.model,
                        arch="xlstm", out_basename=args.out)
    print_report(summary)
    print(f"\nsaved: {args.out}.csv  +  {args.out}.json")


if __name__ == "__main__":
    main()
