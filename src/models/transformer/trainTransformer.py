"""
Train the Transformer tagger on cached (emb, labels).
run:    python -m src.models.transformer.trainTransformer --cache cache/ --epochs 15 --run-dir runs/transformer_run1
"""

import argparse

import torch

from src.models import trainCommon
from src.models.transformer.transformerWithMiniLM import TransformerTagger


def build_model(config):
    return TransformerTagger(
        d_model=config.get("d_model", 128),
        nhead=config.get("nhead", 4),
        num_layers=config.get("num_layers", 4),
        dim_feedforward=config.get("dim_feedforward", 256),
        dropout=config.get("dropout", 0.1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--out", default=None,
                     help="legacy: also copy the best model.pt here for evaluate*.py/tryout.py back-compat")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dim_feedforward", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--clip-grad-norm", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=0, help="early-stop patience on val F1, 0=disabled")
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--resume", default=None, help="path to a checkpoint.pt to resume from")
    ap.add_argument("--run-dir", default=None, help="default: auto-generated under runs/")
    ap.add_argument("--exclude-dataset", default=None,
                     help="drop wceb-<name>-*.npz cache files for leave-one-dataset-out training")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = trainCommon.list_cache_files(args.cache, exclude_dataset=args.exclude_dataset)
    train_files, val_files = trainCommon.train_val_split(files, args.val_frac, args.seed)
    print(f"train={len(train_files)}  val={len(val_files)}  device={device}")

    train_data = trainCommon.load_cache_to_memory(train_files)
    val_data = trainCommon.load_cache_to_memory(val_files)

    config = {
        "arch": "transformer", "d_model": args.d_model, "nhead": args.nhead,
        "num_layers": args.num_layers, "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout, "clip_grad_norm": args.clip_grad_norm,
        "lr": args.lr, "epochs": args.epochs, "val_frac": args.val_frac,
        "seed": args.seed, "cache": args.cache, "exclude_dataset": args.exclude_dataset,
    }
    model = build_model(config)

    run_dir = args.run_dir or trainCommon.make_run_dir("runs", "transformer", config)
    print(f"run_dir={run_dir}")

    trainCommon.train(
        config, model, train_data, val_data, device, run_dir,
        epochs=args.epochs, lr=args.lr, patience=args.patience, min_delta=args.min_delta,
        clip_grad_norm=args.clip_grad_norm, resume_path=args.resume,
    )

    if args.out:
        import shutil
        shutil.copyfile(f"{run_dir}/model.pt", args.out)
        print(f"also copied best model to {args.out} (legacy --out)")


if __name__ == "__main__":
    main()
