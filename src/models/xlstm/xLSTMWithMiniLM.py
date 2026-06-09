"""
  * mLSTM-only (slstm_at=[]). The fast sLSTM CUDA kernel needs Compute Capability >= 8.0;
    the V100 (7.0) and RTX 6000 (7.5) are below that.
  *  a tagging task benefits from
    right-context, so we run one stack forward and one on the reversed sequence and
    concatenate, mirroring the BiLSTM's inductive bias.
  * Input/output projection. The block stack works at `embedding_dim`; we project
    431 -> embedding_dim in, and (2*embedding_dim if bidirectional) -> 1 out.
"""

import torch
import torch.nn as nn

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
)

from src.data.combinedLMEmbedder import embed_document, COMBINED_DIM   # COMBINED_DIM = 431


def _make_mlstm_stack(embedding_dim, num_blocks, num_heads, context_length):
    """An mLSTM-only block stack (no sLSTM -> no CUDA-kernel / Compute-Capability requirement)."""
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4,
                qkv_proj_blocksize=4,
                num_heads=num_heads,          # embedding_dim must be divisible by num_heads
            ),
        ),
        slstm_at=[],                          # <- mLSTM-only
        context_length=context_length,       # must be >= the longest block sequence you feed
        num_blocks=num_blocks,
        embedding_dim=embedding_dim,
    )
    return xLSTMBlockStack(cfg)


class XLSTMTagger(nn.Module):
    def __init__(self, input_dim=COMBINED_DIM, embedding_dim=144,
                 num_blocks=2, num_heads=4, context_length=4608,
                 bidirectional=True):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.bidirectional = bidirectional

        self.in_proj = nn.Linear(input_dim, embedding_dim)
        self.fwd = _make_mlstm_stack(embedding_dim, num_blocks, num_heads, context_length)
        if bidirectional:
            self.bwd = _make_mlstm_stack(embedding_dim, num_blocks, num_heads, context_length)
            self.classifier = nn.Linear(embedding_dim * 2, 1)
        else:
            self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> logits: (batch, seq_len)
        h = self.in_proj(x)
        f = self.fwd(h)
        if self.bidirectional:
            # reverse along time, run the second stack, reverse back, concatenate
            b = torch.flip(self.bwd(torch.flip(h, dims=[1])), dims=[1])
            h = torch.cat([f, b], dim=-1)
        else:
            h = f
        return self.classifier(h).squeeze(-1)


@torch.no_grad()
def predict_document(model, raw_html, threshold=0.5, device="cpu"):
    """raw HTML -> predicted label per block (1=content, 0=boilerplate)."""
    emb = embed_document(raw_html)
    x = torch.as_tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
    probs = torch.sigmoid(model(x))
    return (probs.squeeze(0) > threshold).long().cpu()


if __name__ == "__main__":
    model = XLSTMTagger().to("cpu").eval()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 64, COMBINED_DIM)          # one page, 64 blocks
    y = model(x)
    print(f"XLSTMTagger ready. input_dim={COMBINED_DIM}  out={tuple(y.shape)}  params={n_params:,}")