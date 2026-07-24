"""
Train the xLSTM on cached (emb, labels).
run:    python -m src.models.xlstm.trainxLSTM --cache cache/ --epochs 15 --run-dir runs/xlstm_run1
"""

import argparse

import torch

from src.models import trainCommon
from src.models.xlstm.xLSTMWithMiniLM import XLSTMTagger


def build_model(config):
    return XLSTMTagger(
        embedding_dim=config.get("embedding_dim", 144),
        num_blocks=config.get("num_blocks", 2),
        num_heads=config.get("num_heads", 4),
        context_length=config.get("context_length", 4608),
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
    ap.add_argument("--embedding_dim", type=int, default=144)
    ap.add_argument("--num_blocks", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--context_length", type=int, default=4608)
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
        "arch": "xlstm", "embedding_dim": args.embedding_dim, "num_blocks": args.num_blocks,
        "num_heads": args.num_heads, "context_length": args.context_length,
        "clip_grad_norm": args.clip_grad_norm, "lr": args.lr, "epochs": args.epochs,
        "val_frac": args.val_frac, "seed": args.seed, "cache": args.cache,
        "exclude_dataset": args.exclude_dataset,
    }
    model = build_model(config)

    run_dir = args.run_dir or trainCommon.make_run_dir("runs", "xlstm", config)
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
