"""
Differentiable counterpart to src.data.miniLmEmbedder: the same MiniLM model
(sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), but wired so its ~117.5M
parameters receive gradients from the tagging loss instead of being a frozen, offline,
pre-computed feature extractor.

src.data.miniLmEmbedder's `_model.encode(...)` is decorated `@torch.inference_mode()`
upstream (confirmed by reading the installed sentence_transformers source) -- its output can
never re-enter an autograd graph, no matter what you do with it afterward. That's why this
module bypasses `.encode()` entirely: it pulls out the underlying HF `BertModel`
(`model[0].auto_model`) and tokenizer, and does the forward pass + masked-mean pooling itself,
matching the model's own pooling config (`pooling_mode="mean"`, confirmed via
`SentenceTransformer(...)[1]`, no extra normalization step in the pipeline).

The 47-dim hand-engineered structural features (src.data.combinedLMEmbedder.hand_features)
stay exactly as they are: non-differentiable, computed from the raw BeautifulSoup Tag, and
concatenated onto the now-trainable 384-dim MiniLM output -- same 431-dim COMBINED_DIM shape
every tagger already expects, so nothing downstream needs to change.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from src.data.combinedLMEmbedder import hand_features, COMBINED_DIM

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_LENGTH = 128   # matches the model's own sentence_bert_config.json max_seq_length


class TrainableMiniLMEmbedder(nn.Module):
    def __init__(self, model_name=MODEL_NAME, max_length=MAX_LENGTH):
        super().__init__()
        st_model = SentenceTransformer(model_name)   # used only to obtain the pieces below
        self.auto_model = st_model[0].auto_model      # HF BertModel, a real nn.Module submodule
        self.tokenizer = st_model[0].tokenizer
        self.max_length = max_length

    def forward(self, blocks):
        """blocks: list of parsed bs4.Tag (one page), same input combinedLMEmbedder.embed_blocks
        takes. Returns (len(blocks), COMBINED_DIM) -- differentiable w.r.t. self.auto_model's
        parameters, non-differentiable w.r.t. the hand-feature half."""
        device = next(self.auto_model.parameters()).device
        n = len(blocks)
        if n == 0:
            return torch.zeros((0, COMBINED_DIM), device=device)

        texts = [str(b) for b in blocks]
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                  max_length=self.max_length, return_tensors="pt").to(device)
        out = self.auto_model(**encoded)
        token_embeddings = out.last_hidden_state                       # (n, seq_len, 384)
        mask = encoded["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)  # (n, seq_len, 1)
        summed = (token_embeddings * mask).sum(dim=1)                  # (n, 384)
        counts = mask.sum(dim=1).clamp(min=1e-9)                       # (n, 1)
        pooled = summed / counts                                       # (n, 384), masked mean

        hand = torch.stack([torch.from_numpy(hand_features(b, i, n)) for i, b in enumerate(blocks)])
        hand = hand.to(device=device, dtype=pooled.dtype)

        return torch.cat([pooled, hand], dim=-1)                       # (n, 431)


if __name__ == "__main__":
    from bs4 import BeautifulSoup
    embedder = TrainableMiniLMEmbedder().eval()
    n_params = sum(p.numel() for p in embedder.parameters())
    demo = BeautifulSoup(
        "<div _item_id='0' class='post-content'><h1>Hi</h1>"
        "<p>Some real text here with words.</p></div>", "html.parser"
    ).find("div")
    with torch.no_grad():
        out = embedder([demo])
    print(f"TrainableMiniLMEmbedder ready. params={n_params:,}  out={tuple(out.shape)}")
