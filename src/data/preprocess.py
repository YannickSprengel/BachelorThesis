#!/usr/bin/env python3
"""
preprocess.py  —  run once on the cluster to build train/val/test .jsonl files.

Uses mineru_html's simplify_html and extract_main_html directly.
You only add label induction on top.

Usage:
    python preprocess.py --input data/webmainbench.jsonl --out_dir data/processed/
"""

import argparse
import json
import random
import re
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

# ── the only import from mineru_html ──────────────────────────────────────────
from mineru_html.process.simplify_html import simplify_html, extract_main_html
# ──────────────────────────────────────────────────────────────────────────────

from label_inducer import induce_labels

IGNORE = -100  # PyTorch CrossEntropyLoss ignore index


def process_one(record: dict, tokenizer, max_len: int, max_per_elem: int) -> dict | None:
    raw_html   = record.get("html", "")
    clean_text = record.get("clean_text") or record.get("text", "")
    doc_id     = record.get("url") or record.get("id", "")

    if not raw_html or not clean_text:
        return None

    # ── Step 1: simplify (mineru_html does this) ──────────────────────────────
    # Inspect what simplify_html returns and adjust the key names in
    # label_inducer.py accordingly. It returns a list of element dicts.
    try:
        elements = simplify_html(raw_html)   # list of {item_id, tag, text, ...}
    except Exception:
        return None

    if not elements:
        return None

    # ── Step 2: induce labels (you wrote this) ────────────────────────────────
    labels = induce_labels(elements, clean_text)

    if not any(v == 1 for v in labels.values()):
        return None  # nothing labeled main → skip (bad alignment)

    # ── Step 3: tokenize flat sequence ───────────────────────────────────────
    input_ids, token_labels, attention_mask = [], [], []
    budget = max_len - 2  # reserve for [CLS] and [SEP]

    for elem in elements:
        if budget <= 0:
            break
        text = elem.get("text", "").strip()
        if not text:
            continue

        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=min(max_per_elem, budget),
        )["input_ids"]

        if not ids:
            continue

        label = labels.get(elem["item_id"], None)
        tok_labels = [label if label is not None else IGNORE] * len(ids)

        input_ids.extend(ids)
        token_labels.extend(tok_labels)
        attention_mask.extend([1] * len(ids))
        budget -= len(ids)

    if not input_ids:
        return None

    # Wrap with special tokens
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    input_ids      = [cls] + input_ids      + [sep]
    token_labels   = [IGNORE] + token_labels + [IGNORE]
    attention_mask = [1]    + attention_mask + [1]

    return {
        "doc_id":         doc_id,
        "input_ids":      input_ids,
        "token_labels":   token_labels,
        "attention_mask": attention_mask,
        "original_html":  raw_html,   # needed at inference for extract_main_html
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True,  help="WebMainBench .jsonl")
    parser.add_argument("--out_dir",     required=True,  help="Output directory")
    parser.add_argument("--tokenizer",   default="bert-base-uncased")
    parser.add_argument("--max_len",     type=int, default=512)
    parser.add_argument("--max_per_elem",type=int, default=64)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    # Load and shuffle
    records = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    random.seed(args.seed)
    random.shuffle(records)

    # Split 70/15/15
    n = len(records)
    splits = {
        "train": records[:int(0.7*n)],
        "val":   records[int(0.7*n):int(0.85*n)],
        "test":  records[int(0.85*n):],
    }

    for split_name, split_records in splits.items():
        processed, failed = [], 0
        for rec in tqdm(split_records, desc=split_name):
            result = process_one(rec, tokenizer, args.max_len, args.max_per_elem)
            if result:
                processed.append(result)
            else:
                failed += 1

        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for r in processed:
                f.write(json.dumps(r) + "\n")

        print(f"{split_name}: {len(processed)} ok, {failed} failed → {out_path}")


if __name__ == "__main__":
    main()
