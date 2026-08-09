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

Gradient checkpointing is enabled on auto_model: backprop through a 12-layer BertModel would
otherwise need every layer's activations for every block of a page held in memory at once.
Checkpointing trades some recompute time in the backward pass for not storing those activations,
and only kicks in while self.training is True (HF checks this internally), so eval/inference
(already @torch.no_grad() in predict_page) is unaffected.

Checkpointing alone isn't enough, though: it reduces how much is kept *across* layers, not the
size of any single forward call. forward() used to hand every block of a page to the tokenizer
and auto_model in one batch, so a page with a few hundred blocks still made a single Linear
layer's forward allocate multiple GiB in one shot -- confirmed via an actual OOM traceback
pointing at one `F.linear` call trying to allocate 3 GiB. forward() now chunks blocks into
groups of `chunk_size` before tokenizing/embedding, bounding peak memory per forward call
regardless of how many blocks a page has. Padding is computed per chunk (shorter on average
than one page-wide batch), which doesn't change any block's pooled output: attention_mask
already makes every block's masked-mean pooling ignore padding, whether that padding came from
its own chunk or a larger batch.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from src.data.combinedLMEmbedder import hand_features, COMBINED_DIM

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_LENGTH = 128   # matches the model's own sentence_bert_config.json max_seq_length
CHUNK_SIZE = 16    # blocks per forward call; lower this further if OOMs persist on a given GPU


class TrainableMiniLMEmbedder(nn.Module):
    def __init__(self, model_name=MODEL_NAME, max_length=MAX_LENGTH, chunk_size=CHUNK_SIZE):
        super().__init__()
        st_model = SentenceTransformer(model_name)   # used only to obtain the pieces below
        self.auto_model = st_model[0].auto_model      # HF BertModel, a real nn.Module submodule
        self.tokenizer = st_model[0].tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def _embed_chunk(self, chunk, device):
        texts = [str(b) for b in chunk]
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                  max_length=self.max_length, return_tensors="pt").to(device)
        out = self.auto_model(**encoded)
        token_embeddings = out.last_hidden_state                       # (chunk_n, seq_len, 384)
        mask = encoded["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
        summed = (token_embeddings * mask).sum(dim=1)                  # (chunk_n, 384)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts                                         # (chunk_n, 384), masked mean

    def forward(self, blocks):
        """blocks: list of parsed bs4.Tag (one page), same input combinedLMEmbedder.embed_blocks
        takes. Returns (len(blocks), COMBINED_DIM) -- differentiable w.r.t. self.auto_model's
        parameters, non-differentiable w.r.t. the hand-feature half."""
        device = next(self.auto_model.parameters()).device
        n = len(blocks)
        if n == 0:
            return torch.zeros((0, COMBINED_DIM), device=device)

        pooled = torch.cat([
            self._embed_chunk(blocks[i:i + self.chunk_size], device)
            for i in range(0, n, self.chunk_size)
        ], dim=0)                                                      # (n, 384)

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
