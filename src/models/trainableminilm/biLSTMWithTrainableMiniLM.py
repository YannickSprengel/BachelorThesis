"""
BiLSTM tagger on top of a MiniLM embedder that gets gradients from the tagging loss, instead of
the frozen, precomputed embeddings every other architecture in this repo trains on. Composes two
already-existing, unmodified pieces:

    src.data.trainableMiniLM.TrainableMiniLMEmbedder  (differentiable MiniLM + hand features)
    src.models.lstm.biLSTMWithMiniLM.BiLSTMTagger      (same tagger every frozen-embedding run uses)

Starts with BiLSTM specifically, the cheapest/best-understood tagger in this repo -- swapping in
GRU/xLSTM/Transformer later is just swapping self.tagger, no change needed here.
"""

import torch.nn as nn

from src.data.trainableMiniLM import TrainableMiniLMEmbedder
from src.models.lstm.biLSTMWithMiniLM import BiLSTMTagger


class TrainableMiniLMBiLSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedder = TrainableMiniLMEmbedder()
        self.tagger = BiLSTMTagger(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, blocks):
        """blocks: list[bs4.Tag] for one page. Returns (1, len(blocks)) logits, matching every
        other tagger's forward signature once wrapped with the leading batch dim."""
        emb = self.embedder(blocks).unsqueeze(0)   # (1, seq_len, COMBINED_DIM)
        return self.tagger(emb)


if __name__ == "__main__":
    model = TrainableMiniLMBiLSTM().eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TrainableMiniLMBiLSTM ready. total_params={n_params:,}")
