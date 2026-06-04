"""
train.py
========
Train the BiLSTM on cached (emb, labels). BCEWithLogitsLoss + pos_weight for the
class imbalance, random train/val split, saves the best model by content-F1.

    python train.py --cache cache/ --epochs 15 --out model.pt
"""

import os, glob, argparse, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from bilstm_tagger import BiLSTMTagger


class CachedDocs(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        return torch.from_numpy(d["emb"]), torch.from_numpy(d["labels"])


def pos_weight_from(files):
    pos = neg = 0.0
    for f in files:
        y = np.load(f)["labels"]
        pos += y.sum()
        neg += len(y) - y.sum()
    return torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    for emb, y in loader:
        emb = emb.to(device)                                  # (1, seq, 384)
        pred = torch.sigmoid(model(emb)).squeeze(0).cpu() > 0.5
        y = y.squeeze(0).bool()
        tp += int((pred & y).sum()); fp += int((pred & ~y).sum()); fn += int((~pred & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(os.path.join(args.cache, "*.npz")))
    random.Random(args.seed).shuffle(files)
    n_val = int(len(files) * args.val_frac)
    val_files, train_files = files[:n_val], files[n_val:]
    print(f"train={len(train_files)}  val={len(val_files)}  device={device}")

    train_loader = DataLoader(CachedDocs(train_files), batch_size=1, shuffle=True)
    val_loader   = DataLoader(CachedDocs(val_files),   batch_size=1)

    model = BiLSTMTagger().to(device)
    pos_w = pos_weight_from(train_files).to(device)
    print("pos_weight =", round(pos_w.item(), 3))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for emb, y in train_loader:
            emb, y = emb.to(device), y.to(device)             # (1, seq, 384), (1, seq)
            optimizer.zero_grad()
            loss = criterion(model(emb), y)                   # logits (1, seq) vs y (1, seq)
            loss.backward()
            optimizer.step()
            total += loss.item()
        prec, rec, f1 = evaluate(model, val_loader, device)
        print(f"epoch {epoch:02d}  loss {total/len(train_loader):.4f}  "
              f"val  P {prec:.3f}  R {rec:.3f}  F1 {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.out)
            print(f"  -> saved {args.out} (F1={f1:.3f})")
    print("best val F1:", round(best_f1, 4))


if __name__ == "__main__":
    main()