"""
Usage:
    python test_local.py path/to/page.html
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
    _TAG_VOCAB,
)
from torch.utils.data import DataLoader

N_TAGS  = len(_TAG_VOCAB)                 # 28, NOT 30
POS_IDX = N_TAGS + 8 + 2 + 2             # relative-position feature (== 40)
assert FEATURE_DIM == N_TAGS + 8 + 2 + 2 + 1 + 2 + 4, "feature layout mismatch!"

if len(sys.argv) > 1:
    html = Path(sys.argv[1]).read_text(encoding='utf-8', errors='ignore')
    print(f"Loaded: {sys.argv[1]}")
else:
    # built-in toy page
    html = """
    <html><body>
      <nav class="c-header__nav"><a href="#">Home</a><a href="#">About</a></nav>
      <article class="c-article-body">
        <h1 class="c-article-title">A Short Title</h1>
        <p>This is the first real paragraph of the article. It contains several
           sentences so that the text-statistics features have something to chew on.</p>
        <p>Here is a second content paragraph with more words and punctuation.</p>
      </article>
      <footer class="c-footer"><a href="#">Privacy</a><a href="#">Terms</a></footer>
    </body></html>
    """
    print("Loaded: built-in toy HTML (pass a path to use your own file)")

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
print(f"\n── Feature vectors (FEATURE_DIM = {FEATURE_DIM}, {N_TAGS} tags) ──────────────────────")
for i, block in enumerate(simplified_blocks):
    vec = feature_extractor.extract(block, i, len(simplified_blocks))
    print(f"  block {i+1}: shape={vec.shape}  min={vec.min():.3f}  max={vec.max():.3f}  "
          f"tag_onehot_sum={vec[:N_TAGS].sum():.0f}  rel_pos={vec[POS_IDX]:.2f}")

# Step 3: build dataset
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

print(f"  Your model receives:  (batch, seq_len, {FEATURE_DIM})")
print(f"  Your model outputs:   (batch, seq_len, 2)  → logits per block")