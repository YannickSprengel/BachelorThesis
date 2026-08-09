"""
Train BiLSTM + a jointly finetuned MiniLM embedder on WebMainBench.

Unlike every other train<Arch>.py, this doesn't read a precomputed .npz cache: MiniLM embedding
happens inside the model's forward pass now (see biLSTMWithTrainableMiniLM.TrainableMiniLMBiLSTM),
so training reads webmainbench.jsonl directly and does DOM segmentation + labeling once at
startup (blockDataset.load_webmainbench_blocks), keeping parsed blocks + labels in RAM.

Much more expensive than the cache-based architectures: every step is a full forward+backward
through MiniLM's ~117.5M params, not just a few hundred thousand on precomputed 431-dim vectors.
Not wired into sweep.py/aggregateLODO.py on purpose -- see docs/RESULTS.md and the "trainable
MiniLM" plan notes for why this stays a standalone script for a handful of manually chosen configs.

run:
  python -m src.models.trainableminilm.trainTrainableMiniLM \
      --jsonl src/data/webmainbench.jsonl --epochs 1 --limit 20   # smoke test
  python -m src.models.trainableminilm.trainTrainableMiniLM \
      --jsonl src/data/webmainbench.jsonl --epochs 15             # real run, GPU, std partition
"""

import argparse

import torch

from src.models import trainCommon
from src.models.trainableminilm.blockDataset import load_webmainbench_blocks
from src.models.trainableminilm.biLSTMWithTrainableMiniLM import TrainableMiniLMBiLSTM


def build_model(config):
    return TrainableMiniLMBiLSTM(
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 1),
        dropout=config.get("dropout", 0.3),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="path to webmainbench.jsonl")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--threshold", type=float, default=0.5,
                     help="min fraction of a block's words that must be in the GT vocab")
    ap.add_argument("--min-words", type=int, default=3,
                     help="blocks shorter than this need a full word match, not just threshold")
    ap.add_argument("--limit", type=int, default=0, help="load only first N pages (0 = all); use for smoke tests")
    ap.add_argument("--patience", type=int, default=0, help="early-stop patience on val F1, 0=disabled")
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--clip-grad-norm", type=float, default=1.0,
                     help="finetuning MiniLM end-to-end can be unstable, unlike the small taggers; clip by default")
    ap.add_argument("--resume", default=None, help="path to a checkpoint.pt to resume from")
    ap.add_argument("--run-dir", default=None, help="default: auto-generated under runs/")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    data = load_webmainbench_blocks(args.jsonl, threshold=args.threshold,
                                     min_words=args.min_words, limit=args.limit)
    train_data, val_data = trainCommon.train_val_split(data, args.val_frac, args.seed)
    print(f"train={len(train_data)}  val={len(val_data)}", flush=True)

    config = {
        "arch": "trainable_minilm_bilstm", "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
        "dropout": args.dropout, "lr": args.lr, "epochs": args.epochs, "val_frac": args.val_frac,
        "seed": args.seed, "jsonl": args.jsonl, "threshold": args.threshold, "min_words": args.min_words,
        "clip_grad_norm": args.clip_grad_norm,
    }
    model = build_model(config)

    run_dir = args.run_dir or trainCommon.make_run_dir("runs", "trainable_minilm_bilstm", config)
    print(f"run_dir={run_dir}", flush=True)

    trainCommon.train(
        config, model, train_data, val_data, device, run_dir,
        epochs=args.epochs, lr=args.lr, patience=args.patience, min_delta=args.min_delta,
        clip_grad_norm=args.clip_grad_norm, resume_path=args.resume,
        # blocks stay an unbatched list (the model adds the batch dim itself, see
        # TrainableMiniLMBiLSTM.forward); the label tensor needs the leading batch dim of 1
        # that torch's default collate would add for a tensor, to match model(blocks)'s output.
        collate_fn=lambda batch: (batch[0][0], batch[0][1].unsqueeze(0)),
    )


if __name__ == "__main__":
    main()
