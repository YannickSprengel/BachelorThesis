"""
bilstm_tagger.py
================
BiLSTM that consumes the (seq_len, 384) MiniLM block embeddings from
minilm_embedder.embed_document and predicts ONE sigmoid score per block.

    raw HTML --embed_document--> (seq_len, 384) --BiLSTM--> (seq_len,) logit --sigmoid--> P(content)
"""

import torch
import torch.nn as nn

from src.data.miniLmEmbedder import embed_document, EMB_DIM   # EMB_DIM = 384


class BiLSTMTagger(nn.Module):
    def __init__(self, input_dim=EMB_DIM, hidden_dim=128,
                 num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, 1)   # *2 bidirectional, 1 logit per block

    def forward(self, x):
        # x: (batch, seq_len, 384) -> logits: (batch, seq_len)   [raw, pre-sigmoid]
        out, _ = self.lstm(x)
        return self.classifier(out).squeeze(-1)


@torch.no_grad()
def predict_document(model, raw_html, threshold=0.5, device="cpu"):
    """raw HTML -> predicted label per block (1=content, 0=boilerplate)."""
    emb = embed_document(raw_html)                                 # (seq_len, 384) numpy
    x = torch.as_tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)  # (1, seq_len, 384)
    probs = torch.sigmoid(model(x))                               # (1, seq_len) in [0,1]
    return (probs.squeeze(0) > threshold).long().cpu()            # (seq_len,)


if __name__ == "__main__":
    model = BiLSTMTagger().to("cpu").eval()
    # preds = predict_document(model, raw_html)
    print("BiLSTMTagger ready (sigmoid head). input_dim =", EMB_DIM)