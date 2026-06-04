"""
try_it.py
=========
Load a trained model, extract main content from ONE HTML file, and reconstruct it
from the MAPPING branch (the part that keeps the original DOM structure).

    python try_it.py --model model.pt --html page.html
"""

import argparse
import torch
from src.data.miniLmEmbedder import simplify_html, _parse_blocks, embed_document
from src.models.lstm.biLSTMWithMiniLM import BiLSTMTagger


@torch.no_grad()
def extract_main_content(model, html, threshold=0.5, device="cpu"):
    simplified_html_str, mapping_html_str = simplify_html(html)
    simpl   = _parse_blocks(simplified_html_str)   # what the model sees
    mapping = _parse_blocks(mapping_html_str)       # original DOM, used to rebuild content
    emb = torch.as_tensor(embed_document(simpl), dtype=torch.float32, device=device).unsqueeze(0)
    keep = (torch.sigmoid(model(emb)).squeeze(0).cpu() > threshold)
    # reconstruct from the mapping branch wherever the block was predicted "content"
    selected = [str(b) for b, k in zip(mapping, keep) if k]
    return "\n".join(selected), keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--html", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BiLSTMTagger().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    html = open(args.html, encoding="utf-8", errors="ignore").read()
    main_html, keep = extract_main_content(model, html, device=device)
    print(f"kept {int(keep.sum())}/{len(keep)} blocks as main content\n")
    print(main_html[:3000])


if __name__ == "__main__":
    main()