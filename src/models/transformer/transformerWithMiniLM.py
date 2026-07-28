"""
Transformer encoder over the sequence of per-block embeddings for one page: full
bidirectional self-attention across all blocks (no causal mask -- a tagging task benefits
from full context in both directions, same reason BiLSTM runs both directions and xLSTM
runs a forward + flipped-input stack). batch_size is always 1 here (whole documents as
variable-length sequences), so no padding/attention-mask machinery is needed.

Input/output: in_proj 431 -> d_model, sinusoidal positional encoding (block order is
meaningful -- it's the document's reading order), N TransformerEncoderLayers, classifier
d_model -> 1 logit per block. Same predict_document(...) signature as BiLSTMTagger/
XLSTMTagger/BiGRUTagger.

Sized (d_model=128, nhead=4, num_layers=4, dim_feedforward=256) to land roughly in the same
param-count ballpark as BiLSTMTagger (~574,721) so the quality-vs-cost sweep comparison
isn't confounded by one model just being bigger; exact count printed by __main__ below.
"""

import math

import torch
import torch.nn as nn

from src.data.combinedLMEmbedder import embed_document, COMBINED_DIM   # COMBINED_DIM = 431


class PositionalEncoding(nn.Module):
    """Sinusoidal, computed for the actual seq_len on each call -- no fixed max length,
    since block-sequence length varies per page and isn't capped elsewhere in this
    pipeline (unlike xLSTM's context_length)."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device
        pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
                         * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return x + pe.unsqueeze(0)


class TransformerTagger(nn.Module):
    def __init__(self, input_dim=COMBINED_DIM, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.pos_enc(self.in_proj(x))
        h = self.encoder(h)
        return self.classifier(h).squeeze(-1)


@torch.no_grad()
def predict_document(model, raw_html, threshold=0.5, device="cpu"):
    """raw HTML -> predicted label per block (1=content, 0=boilerplate)."""
    emb = embed_document(raw_html)
    x = torch.as_tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
    probs = torch.sigmoid(model(x))
    return (probs.squeeze(0) > threshold).long().cpu()


if __name__ == "__main__":
    model = TransformerTagger().to("cpu").eval()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 64, COMBINED_DIM)          # one page, 64 blocks
    y = model(x)
    print(f"TransformerTagger ready. input_dim={COMBINED_DIM}  out={tuple(y.shape)}  params={n_params:,}"
          f"  (target ~574,721, BiLSTM's count)")
