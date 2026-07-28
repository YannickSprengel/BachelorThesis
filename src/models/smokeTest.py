"""
Fast local integration check: for every architecture registered in sweep.py's
ARCH_BUILDERS, build the default model and train it for 1 epoch against a small cache,
asserting it completes without an exception or a NaN loss and that model.pt/config.json/
metrics.json actually get written. Not a pytest suite -- this repo has no test framework
configured and this script doesn't introduce one, it's a couple-minutes-locally gate before
spending cluster time on a real sweep (this is exactly the kind of integration break that
silently ate a run before: a bad node's CUDA compute capability or a malformed cache file
surfaces here in seconds, not after an hours-long cluster job).

run: python -m src.models.smokeTest --cache combined_cache/
"""

import argparse
import json
import math
import os
import shutil
import time

import torch

from src.models import trainCommon
from src.models.sweep import ARCH_BUILDERS, ARCH_DEFAULTS


def run_one(arch, cache_dir, out_dir, val_frac, seed, device, epochs):
    build_model = ARCH_BUILDERS[arch]
    defaults = ARCH_DEFAULTS[arch]
    config = {"arch": arch, **defaults, "epochs": epochs, "val_frac": val_frac, "seed": seed,
              "cache": cache_dir}

    files = trainCommon.list_cache_files(cache_dir)
    if len(files) < 2:
        return {"arch": arch, "status": "FAIL", "reason": f"cache has only {len(files)} files, need >=2"}

    train_files, val_files = trainCommon.train_val_split(files, val_frac, seed)
    if not train_files or not val_files:
        return {"arch": arch, "status": "FAIL", "reason": "empty train or val split, use a bigger cache"}

    train_data = trainCommon.load_cache_to_memory(train_files)
    val_data = trainCommon.load_cache_to_memory(val_files)

    run_dir = os.path.join(out_dir, arch)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    t0 = time.time()
    try:
        model = build_model(config)
        metrics = trainCommon.train(
            config, model, train_data, val_data, device, run_dir,
            epochs=epochs, lr=config["lr"], patience=0, clip_grad_norm=config.get("clip_grad_norm"),
        )
    except Exception as e:
        return {"arch": arch, "status": "FAIL", "reason": f"{type(e).__name__}: {e}",
                "sec": round(time.time() - t0, 1)}

    losses = [h["train_loss"] for h in metrics["history"]]
    if any(math.isnan(v) or math.isinf(v) for v in losses):
        return {"arch": arch, "status": "FAIL", "reason": f"non-finite train_loss in {losses}",
                "sec": round(time.time() - t0, 1)}

    # checkpoint.pt is written unconditionally every epoch; model.pt only on a val-F1
    # improvement over the 0.0 baseline, which a couple of epochs on a tiny cache isn't
    # guaranteed to produce -- require the former (proves the loop ran end to end), not
    # the latter (would conflate "crashed" with "didn't beat 0.0 yet").
    for fname in ("checkpoint.pt", "config.json", "metrics.json"):
        if not os.path.exists(os.path.join(run_dir, fname)):
            return {"arch": arch, "status": "FAIL", "reason": f"missing {fname} in {run_dir}",
                    "sec": round(time.time() - t0, 1)}
    train_loss = losses[-1]

    return {"arch": arch, "status": "PASS", "train_loss": round(train_loss, 4),
            "val_f1": round(metrics["best_val_f1"], 4), "sec": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="combined_cache",
                     help="small cache dir to smoke-test against (not the full training cache)")
    ap.add_argument("--out", default="runs/smoke_test")
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2,
                    help="kept small (this is a crash/NaN check, not a convergence check) but >1 "
                         "so slower-to-learn architectures (e.g. Transformer) get more than one "
                         "gradient step before we look at their output")
    ap.add_argument("--arch", nargs="*", default=None,
                    help="subset of architectures to test (default: all registered in sweep.py)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    archs = args.arch or list(ARCH_BUILDERS)
    print(f"smoke-testing {archs} on cache={args.cache}  device={device}  epochs={args.epochs}\n")

    results = [run_one(arch, args.cache, args.out, args.val_frac, args.seed, device, args.epochs)
               for arch in archs]

    print(f"\n{'arch':<12} {'status':<6} {'val_f1':>8} {'train_loss':>11} {'sec':>6}  reason")
    n_fail = 0
    for r in results:
        if r["status"] == "FAIL":
            n_fail += 1
            print(f"{r['arch']:<12} {r['status']:<6} {'':>8} {'':>11} {r.get('sec', ''):>6}  {r['reason']}")
        else:
            print(f"{r['arch']:<12} {r['status']:<6} {r['val_f1']:>8} {r['train_loss']:>11} {r['sec']:>6}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "smoke_test.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if n_fail:
        print(f"\n{n_fail}/{len(results)} architectures FAILED. Fix before spending cluster time.")
        raise SystemExit(1)
    print(f"\nall {len(results)} architectures passed.")


if __name__ == "__main__":
    main()
