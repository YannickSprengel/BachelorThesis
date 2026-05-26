"""
test_pipeline.py
================
Quick smoke-test for dripper_lstm_pipeline.py against a local HTML file.

Usage:
    python test_pipeline.py                        # uses built-in toy HTML
    python test_pipeline.py path/to/page.html      # uses your own file
"""

import sys
from pathlib import Path

# ── import the pipeline ────────────────────────────────────────────────────────
from preprocess import (
    DripperPreprocessor,
    BlockFeatureExtractor,
    LabelGenerator,
    HTMLExtractionDataset,
    collate_fn,
    FEATURE_DIM,
)
from torch.utils.data import DataLoader

# ── load HTML ─────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    html = Path(sys.argv[1]).read_text(encoding='utf-8', errors='ignore')
    print(f"Loaded: {sys.argv[1]}")
preprocessor      = DripperPreprocessor()
feature_extractor = BlockFeatureExtractor()
label_generator   = LabelGenerator()

# Step 1: simplify_html → blocks
simplified_blocks, mapping_blocks = preprocessor.process(html)

print(f"\n── Blocks ({len(simplified_blocks)} total) ──────────────────────────────")
for block in simplified_blocks:
    item_id  = block.get('_item_id', '?')
    tag      = block.name
    text     = block.get_text(separator=' ', strip=True)[:60]
    cls      = block.get('class', [])
    bid      = block.get('id', '')
    print(f"  [{item_id:>3}] <{tag}> class={cls} id={repr(bid):<15}  text: {repr(text)}")

# Step 2: extract features
print(f"\n── Feature vectors (FEATURE_DIM = {FEATURE_DIM}) ──────────────────────")
for i, block in enumerate(simplified_blocks):
    vec = feature_extractor.extract(block, i, len(simplified_blocks))
    print(f"  block {i+1}: shape={vec.shape}  min={vec.min():.3f}  max={vec.max():.3f}  "
          f"tag_onehot_sum={vec[:30].sum():.0f}  rel_pos={vec[42]:.2f}")

# Step 3: build dataset (no ground truth → unlabelled mode)
dataset = HTMLExtractionDataset(preprocessor, feature_extractor, label_generator)
dataset.add_document(raw_html=html, label_format='unlabelled')

features, labels = dataset[0]
print(f"\n── Dataset sample ──────────────────────────────────────────────────────")
print(f"  features tensor : {features.shape}   dtype={features.dtype}")
print(f"  labels tensor   : {labels.shape}    dtype={labels.dtype}  (all -1 = unlabelled)")

# Step 4: DataLoader
loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
for feat_batch, lbl_batch, lengths, mask in loader:
    print(f"\n── DataLoader batch ────────────────────────────────────────────────────")
    print(f"  feat_batch : {feat_batch.shape}   → (batch=1, seq_len, {FEATURE_DIM})")
    print(f"  lbl_batch  : {lbl_batch.shape}")
    print(f"  lengths    : {lengths.tolist()}")
    print(f"  mask       : {mask.shape}  — {mask.sum().item()} real positions")

print("\n✓ Pipeline working. Ready to plug in your BiLSTM.")
print(f"  Your model receives:  (batch, seq_len, {FEATURE_DIM})")
print(f"  Your model outputs:   (batch, seq_len, 2)  → logits per block")