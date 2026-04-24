import torch
import torch.nn as nn


class LSTMBoilerplateClassifier(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 128,
            hidden_dim: int = 256,
            num_layers: int = 2,
            num_tags: int = 64,  # number of unique HTML tags
            tag_embed_dim: int = 16,
            structural_dim: int = 4,  # depth, link_density, position, text_len
            dropout: float = 0.3,
            bidirectional: bool = True,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.tag_embedding = nn.Embedding(num_tags, tag_embed_dim)

        lstm_input_dim = embed_dim + tag_embed_dim + structural_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * direction_factor, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # binary: content vs boilerplate
        )

    def forward(self, token_ids, tag_ids, structural_feats, lengths):
        # token_ids: (batch, seq_len)
        # tag_ids: (batch, seq_len)
        # structural_feats: (batch, seq_len, structural_dim)
        # lengths: (batch,) actual lengths for packing

        tok_emb = self.token_embedding(token_ids)  # (B, L, embed_dim)
        tag_emb = self.tag_embedding(tag_ids)  # (B, L, tag_embed_dim)

        x = torch.cat([tok_emb, tag_emb, structural_feats], dim=-1)

        # Pack for efficiency with variable-length sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        logits = self.classifier(output).squeeze(-1)  # (B, L)
        return logits  # use BCEWithLogitsLoss during training