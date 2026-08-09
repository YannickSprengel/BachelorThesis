"""
Shared training core for trainLSTM.py / trainxLSTM.py / sweep.py / aggregateLODO.py.

Cache files are .npz per page: {"emb": (n_blocks, COMBINED_DIM), "labels": (n_blocks,)}.
WCEB-sourced files (see cacheEmbeddingsFromWCEB.py) are named "wceb-<dataset>-<page_id>.npz";
WebMainBench-sourced files have no such prefix. list_cache_files()'s exclude_dataset filters
on that prefix, which is what makes leave-one-dataset-out (LODO) training possible without a
separate manifest file.
"""

import os
import glob
import json
import time
import random
import hashlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def list_cache_files(cache_dir, exclude_dataset=None):
    files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    if exclude_dataset:
        prefix = f"wceb-{exclude_dataset}-"
        files = [f for f in files if not os.path.basename(f).startswith(prefix)]
    return files


def train_val_split(files, val_frac, seed):
    files = list(files)
    random.Random(seed).shuffle(files)
    n_val = int(len(files) * val_frac)
    val_files, train_files = files[:n_val], files[n_val:]
    return train_files, val_files


def load_cache_to_memory(files):
    """np.load each file once; returns a list of (emb, labels) tensor pairs kept in RAM.
    Avoids re-reading disk every epoch (the original CachedDocs.__getitem__ called
    np.load on every access) and lets a sweep reuse one load across many configs."""
    data = []
    for f in files:
        d = np.load(f)
        data.append((torch.from_numpy(d["emb"]), torch.from_numpy(d["labels"])))
    return data


class InMemoryDocs(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def pos_weight_from(data):
    pos = neg = 0.0
    for _, y in data:
        pos += float(y.sum())
        neg += float(y.numel() - y.sum())
    return torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)


@torch.no_grad()
def evaluate_block_f1(model, loader, device, threshold=0.5):
    model.eval()
    tp = fp = fn = 0
    for emb, y in loader:
        # emb is a raw block list (not a tensor) for the trainable-MiniLM path; the embedder
        # inside that model reads its own device from its parameters instead.
        emb = emb.to(device) if torch.is_tensor(emb) else emb
        pred = torch.sigmoid(model(emb)).squeeze(0).cpu() > threshold
        y = y.squeeze(0).bool()
        tp += int((pred & y).sum()); fp += int((pred & ~y).sum()); fn += int((~pred & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def make_run_dir(base, arch, config):
    ts = time.strftime("%Y%m%d-%H%M%S")
    h = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    run_dir = os.path.join(base, f"{ts}_{arch}_{h}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_checkpoint(path, model, optimizer, epoch, best_f1, config, history):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_f1": best_f1,
        "config": config,
        "history": history,
    }, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def train(config, model, train_data, val_data, device, run_dir,
          epochs=15, lr=1e-3, patience=0, min_delta=0.0,
          clip_grad_norm=None, resume_path=None, val_threshold=0.5, collate_fn=None):
    """Core training loop, architecture-agnostic (model is already constructed).

    Writes run_dir/config.json (once), run_dir/model.pt (best state_dict, on val-F1
    improvement), run_dir/checkpoint.pt (full resume state, every epoch), and
    run_dir/metrics.json (per-epoch history + summary, every epoch). Returns the
    final metrics dict (same content as metrics.json).

    collate_fn defaults to None (torch's default, stacking tensors) for the cache-based
    architectures. The trainable-MiniLM path passes its own collate_fn instead, since its
    dataset yields (list[bs4.Tag], label_tensor) pairs where torch's default collate can't
    stack the block list -- it unwraps the block list as-is (batch_size is always 1 in this
    repo) but still adds the leading batch dim to the label tensor by hand, to match the
    dim torch's default collate would have added for a plain tensor.
    """
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    model = model.to(device)
    train_loader = DataLoader(InMemoryDocs(train_data), batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(InMemoryDocs(val_data), batch_size=1, collate_fn=collate_fn)

    pos_w = pos_weight_from(train_data).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model_path = os.path.join(run_dir, "model.pt")
    checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
    metrics_path = os.path.join(run_dir, "metrics.json")

    start_epoch = 0
    best_f1 = 0.0
    epochs_since_improve = 0
    history = []

    if resume_path:
        ckpt = load_checkpoint(resume_path, model, optimizer, device)
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        history = ckpt.get("history", [])
        print(f"resumed from {resume_path} at epoch {start_epoch}, best_f1={best_f1:.4f}", flush=True)

    t0 = time.time()
    early_stopped = False
    for epoch in range(start_epoch, epochs):
        model.train()
        total = 0.0
        for emb, y in train_loader:
            emb = emb.to(device) if torch.is_tensor(emb) else emb
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(emb), y)
            loss.backward()
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            total += loss.item()
        train_loss = total / max(len(train_loader), 1)
        prec, rec, f1 = evaluate_block_f1(model, val_loader, device, val_threshold)
        print(f"epoch {epoch:02d}  loss {train_loss:.4f}  "
              f"val  P {prec:.3f}  R {rec:.3f}  F1 {f1:.3f}", flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_p": prec, "val_r": rec, "val_f1": f1})

        if f1 > best_f1 + min_delta:
            best_f1 = f1
            epochs_since_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"  -> saved {model_path} (F1={f1:.3f})", flush=True)
        else:
            epochs_since_improve += 1

        save_checkpoint(checkpoint_path, model, optimizer, epoch, best_f1, config, history)

        best_epoch = max(range(len(history)), key=lambda i: history[i]["val_f1"])
        metrics = {
            "history": history,
            "best_epoch": history[best_epoch]["epoch"],
            "best_val_f1": best_f1,
            "early_stopped": False,
            "wall_time_sec": time.time() - t0,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        if patience > 0 and epochs_since_improve >= patience:
            early_stopped = True
            print(f"  early stopping at epoch {epoch} (patience={patience})", flush=True)
            break

    metrics["early_stopped"] = early_stopped
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("best val F1:", round(best_f1, 4), flush=True)
    return metrics
