"""
run: python -m src.evaluation.aggregateLODO --arch bilstm --cache cache/ \
        --wceb src/evaluation/web-content-extraction-benchmark/datasets/combined \
        --config runs/sweep_bilstm_.../<winning_run>/config.json --out runs/lodo_bilstm

8-fold leave-one-dataset-out (LODO): trains one FIXED hyperparameter config (the winner
from a sweep.py run) 8 times, each time excluding one WCEB sub-dataset from training and
evaluating only on that held-out sub-dataset, so training never sees the exact pages it's
evaluated on. Reuses trainCommon.train + evalCommon.run_eval -- the same "train then
run_eval" pattern sweep.py's top-K stage uses, not a divergent implementation.

Do not fold this into sweep.py's --grid axes: multiplying a sweep's configs by 8 folds
isn't realistic on a shared cluster, and if the sweep's own top-K WCEB eval used a model
trained on any WCEB sub-dataset, that eval would leak against its own training data.
"""

import argparse
import json
import os
import statistics

import torch

from src.models import trainCommon
from src.evaluation.evalCommon import run_eval
from src.models.lstm.trainLSTM import build_model as build_bilstm
from src.models.xlstm.trainxLSTM import build_model as build_xlstm
from src.models.gru.trainGRU import build_model as build_gru
from src.models.transformer.trainTransformer import build_model as build_transformer

ARCH_BUILDERS = {
    "bilstm": build_bilstm, "xlstm": build_xlstm,
    "gru": build_gru, "transformer": build_transformer,
}
WCEB_DATASETS = ["cetd", "cleaneval", "cleanportaleval", "dragnet",
                 "google-trends-2017", "l3s-gn1", "readability", "scrapinghub"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(ARCH_BUILDERS))
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--wceb", required=True, help="path to .../datasets/combined")
    ap.add_argument("--config", required=True, help="config.json of the winning hyperparameters")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", nargs="*", default=None, help="folds to run (default: all 8)")
    ap.add_argument("--out", required=True, help="output dir for per-fold run subdirs + aggregate.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    build_model = ARCH_BUILDERS[args.arch]

    with open(args.config) as f:
        base_config = json.load(f)

    folds = args.datasets or WCEB_DATASETS
    os.makedirs(args.out, exist_ok=True)

    per_fold = []
    for fold in folds:
        print(f"\n=== LODO fold: held out {fold} ===")
        config = {
            **base_config, "arch": args.arch, "exclude_dataset": fold,
            "epochs": args.epochs, "val_frac": args.val_frac, "seed": args.seed,
            "cache": args.cache,
        }

        files = trainCommon.list_cache_files(args.cache, exclude_dataset=fold)
        train_files, val_files = trainCommon.train_val_split(files, args.val_frac, args.seed)
        print(f"train={len(train_files)}  val={len(val_files)}  device={device}")
        train_data = trainCommon.load_cache_to_memory(train_files)
        val_data = trainCommon.load_cache_to_memory(val_files)

        model = build_model(config)
        run_dir = os.path.join(args.out, f"lodo_{args.arch}_{fold}")
        metrics = trainCommon.train(
            config, model, train_data, val_data, device, run_dir,
            epochs=args.epochs, lr=config.get("lr", 1e-3), patience=args.patience,
            min_delta=args.min_delta, clip_grad_norm=config.get("clip_grad_norm"),
        )

        eval_model = build_model(config).to(device)
        eval_model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location=device))
        eval_model.eval()
        summary = run_eval(eval_model, args.wceb, datasets=[fold], device=device,
                            model_path=os.path.join(run_dir, "model.pt"), arch=args.arch,
                            out_basename=os.path.join(run_dir, "wceb"))

        per_fold.append({
            "fold": fold, "val_f1": round(metrics["best_val_f1"], 4),
            "rouge5": summary["rouge5"]["overall_f1"], "rouge_l": summary["rouge_l"]["overall_f1"],
            "block_f1": summary["block_level"]["f1"], "n_params": summary["n_params"],
            "pages_per_sec": summary["throughput"]["pages_per_sec"], "run_dir": run_dir,
        })
        print(f"  fold {fold}: rouge5={per_fold[-1]['rouge5']:.4f}  "
              f"rougeL={per_fold[-1]['rouge_l']:.4f}  block_f1={per_fold[-1]['block_f1']:.4f}")

    def agg(key):
        vals = [r[key] for r in per_fold]
        return {"mean": round(statistics.mean(vals), 4),
                "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0}

    aggregate = {
        "arch": args.arch, "config": args.config, "folds": [r["fold"] for r in per_fold],
        "per_fold": per_fold,
        "rouge5": agg("rouge5"), "rouge_l": agg("rouge_l"), "block_f1": agg("block_f1"),
    }
    out_path = os.path.join(args.out, "aggregate.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n=== LODO aggregate over {len(per_fold)} folds ===")
    print(f"rouge5   mean={aggregate['rouge5']['mean']:.4f}  std={aggregate['rouge5']['std']:.4f}")
    print(f"rougeL   mean={aggregate['rouge_l']['mean']:.4f}  std={aggregate['rouge_l']['std']:.4f}")
    print(f"block_f1 mean={aggregate['block_f1']['mean']:.4f}  std={aggregate['block_f1']['std']:.4f}")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
