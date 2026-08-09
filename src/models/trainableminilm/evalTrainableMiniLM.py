"""
Benchmark a trained TrainableMiniLMBiLSTM on WCEB. Same three things every other evaluate*.py
reports (ROUGE-5/L, block-level P/R/F1, throughput), but the embedding step uses this model's
own finetuned MiniLM weights instead of the frozen combinedLMEmbedder every other architecture
evaluates against -- see evalCommon.run_eval's embed_fn parameter.

model.pt holds the FULL joint model's state dict (embedder + tagger together, since that's what
trainTrainableMiniLM.py passes to trainCommon.train()). Only model.tagger is passed to run_eval
as "the model" doing the emb->logits forward pass; model.embedder is passed as embed_fn so the
already-embedded tensor (not raw blocks) is what predict_page's model(emb) call receives -- same
contract every cache-based architecture's tagger expects.

run: python -m src.models.trainableminilm.evalTrainableMiniLM --model runs/.../model.pt \
        --wceb src/evaluation/wceb_data/combined --out results/trainable_minilm/wceb
"""

import argparse

import torch

from src.evaluation.evalCommon import run_eval, print_report
from src.models.trainableminilm.biLSTMWithTrainableMiniLM import TrainableMiniLMBiLSTM


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

    model = TrainableMiniLMBiLSTM().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  TrainableMiniLMBiLSTM params={n_params:,} (embedder+tagger)")
    print(f"keep-threshold={args.threshold}  label-threshold(silver)={args.label_threshold}")
    print(f"(ROUGE-{args.n} F1 + ROUGE-L F1, jieba  +  block-level P/R/F1  +  throughput)")

    summary = run_eval(model.tagger, args.wceb, datasets=args.datasets, n=args.n, threshold=args.threshold,
                        label_threshold=args.label_threshold, device=device, model_path=args.model,
                        arch="trainable_minilm_bilstm", out_basename=args.out,
                        embed_fn=model.embedder, n_params=n_params)
    print_report(summary)
    print(f"\nsaved: {args.out}.csv  +  {args.out}.json")


if __name__ == "__main__":
    main()
