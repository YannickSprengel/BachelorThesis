"""
BiLSTM block tagger for the Dripper-style HTML main-content-extraction task.

Designed to plug directly into the pipeline in preprocess.py:
    - input : (batch, seq_len, FEATURE_DIM)  feature matrices  (FEATURE_DIM = 49)
    - output: (batch, seq_len, 2)            per-block logits (0=boilerplate, 1=content)

"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

FEATURE_DIM = 47
NUM_CLASSES = 2
PAD_LABEL = -100  # must match collate_fn's labels padding_value


# MODEL
class BiLSTMBlockTagger(nn.Module):
    def __init__(self,
                 input_dim: int = FEATURE_DIM,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = NUM_CLASSES,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            # NOTE: nn.LSTM's internal dropout only fires between stacked layers,
            # so it is a no-op when num_layers == 1.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # *2 because the sequence is read in both directions.
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        features: (B, T, input_dim)
        lengths:  (B,) real sequence lengths (must be on CPU for packing)
        returns:  (B, T, num_classes) logits
        """
        # Pack so the (backward) LSTM never ingests padded timesteps.
        # enforce_sorted=False because collate_fn does not sort the batch by length.
        packed = pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)

        # total_length pins the restored length to T so it matches labels/mask shapes.
        out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=features.size(1)
        )
        out = self.dropout(out)
        return self.classifier(out)  # (B, T, num_classes)


# =============================================================================
# CLASS WEIGHTS (handle boilerplate/content imbalance)
# =============================================================================
def compute_class_weights(dataset, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    Inverse-frequency class weights from the *training* dataset.
    Pass these to CrossEntropyLoss(weight=...) so the minority content class counts more.
    """
    all_labels = np.concatenate([s["labels"] for s in dataset._samples])
    all_labels = all_labels[all_labels != PAD_LABEL]  # ignore any pad sentinels
    counts = np.bincount(all_labels, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)               # avoid div-by-zero
    weights = counts.sum() / (num_classes * counts)   # balanced weighting
    return torch.tensor(weights, dtype=torch.float32)


# =============================================================================
# TRAIN / EVAL LOOPS
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss, total_tokens = 0.0, 0
    for features, labels, lengths, mask in loader:
        features, labels = features.to(device), labels.to(device)
        # lengths stays on CPU for packing.

        optimizer.zero_grad()
        logits = model(features, lengths)                       # (B, T, C)
        # CrossEntropyLoss ignores PAD_LABEL automatically -> no manual masking needed.
        loss = criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # LSTMs benefit from clipping
        optimizer.step()

        n = int(mask.sum().item())
        total_loss += loss.item() * n
        total_tokens += n
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Returns precision/recall/F1 for the CONTENT class (1), computed only on real blocks."""
    model.eval()
    tp = fp = fn = tn = 0
    for features, labels, lengths, mask in loader:
        features = features.to(device)
        logits = model(features, lengths)                 # (B, T, C)
        preds = logits.argmax(dim=-1).cpu()               # (B, T)

        m = mask.cpu().bool()
        p = preds[m]
        y = labels[m]                                     # already excludes pad via mask

        tp += int(((p == 1) & (y == 1)).sum())
        fp += int(((p == 1) & (y == 0)).sum())
        fn += int(((p == 0) & (y == 1)).sum())
        tn += int(((p == 0) & (y == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # from preprocess import HTMLExtractionDataset, FeatureNormalizer, collate_fn

    device = "cuda" if torch.cuda.is_available() else "cpu"


    model = BiLSTMBlockTagger().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)  # add weight=... in real runs
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("Model built:", model)