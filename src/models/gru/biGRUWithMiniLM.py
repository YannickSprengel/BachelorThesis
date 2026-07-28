"""
Bidirectional GRU that consumes the (seq_len, 431) combined block embeddings and predicts
ONE sigmoid score per block. Same interface as BiLSTMTagger/XLSTMTagger: forward(x) ->
(batch, seq_len) logits, predict_document(model, raw_html, threshold, device).

GRU has 3 gates vs. LSTM's 4 (no separate cell state), so at the same hidden_dim it's a
naturally cheaper/faster alternative -- deliberately NOT capacity-matched to BiLSTM here,
since that size difference is itself a data point for the thesis's quality-vs-cost
comparison, not something to normalize away.
"""

import torch
import torch.nn as nn

from src.data.combinedLMEmbedder import embed_document, COMBINED_DIM   # COMBINED_DIM = 431


class BiGRUTagger(nn.Module):
    def __init__(self, input_dim=COMBINED_DIM, hidden_dim=128,
                 num_layers=1, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, 1)   # *2 bidirectional, 1 logit per block

    def forward(self, x):
        out, _ = self.gru(x)
        return self.classifier(out).squeeze(-1)


@torch.no_grad()
def predict_document(model, raw_html, threshold=0.5, device="cpu"):
    """raw HTML -> predicted label per block (1=content, 0=boilerplate)."""
    emb = embed_document(raw_html)                                 # (seq_len, 431) numpy
    x = torch.as_tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)  # (1, seq_len, 431)
    probs = torch.sigmoid(model(x))                               # (1, seq_len) in [0,1]
    return (probs.squeeze(0) > threshold).long().cpu()            # (seq_len,)


if __name__ == "__main__":
    model = BiGRUTagger().to("cpu").eval()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 64, COMBINED_DIM)          # one page, 64 blocks
    y = model(x)
    print(f"BiGRUTagger ready. input_dim={COMBINED_DIM}  out={tuple(y.shape)}  params={n_params:,}")
