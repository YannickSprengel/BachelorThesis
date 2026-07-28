"""
Load a trained model, extract main content from ONE HTML file, and write the FULL
extracted content to disk:
  - <out>.html : self-contained HTML (reconstructed from the MAPPING branch)
  - <out>.md   : Markdown via html2text

 run: python -m src.models.transformer.tryout --model model.pt --html page.html
"""

import argparse
import os
import torch
from src.data.combinedLMEmbedder import simplify_html, _parse_blocks, embed_blocks
from src.models.transformer.transformerWithMiniLM import TransformerTagger


@torch.no_grad()
def extract_main_content(model, html, threshold=0.5, device="cpu"):
    simplified_html_str, mapping_html_str = simplify_html(html)
    simpl   = _parse_blocks(simplified_html_str)   # what the model sees
    mapping = _parse_blocks(mapping_html_str)       # original DOM, used to rebuild content
    emb = torch.as_tensor(embed_blocks(simpl), dtype=torch.float32, device=device).unsqueeze(0)
    keep = (torch.sigmoid(model(emb)).squeeze(0).cpu() > threshold)
    # reconstruct from the mapping branch wherever the block was predicted "content"
    selected = [str(b) for b, k in zip(mapping, keep) if k]
    return "\n".join(selected), keep


def to_markdown(body_html):
    try:
        import html2text
    except ImportError:
        return None
    h = html2text.HTML2Text()
    h.body_width = 0          # don't hard-wrap lines
    return h.handle(body_html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--html", required=True)
    ap.add_argument("--out", default=None, help="output basename (default: <input>.extracted)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TransformerTagger().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    html = open(args.html, encoding="utf-8", errors="ignore").read()
    body, keep = extract_main_content(model, html, device=device)

    out_base = args.out or (os.path.splitext(args.html)[0] + ".extracted")

    # full, self-contained HTML document
    html_doc = (
        '<!DOCTYPE html>\n<html><head><meta charset="utf-8"></head>\n'
        f"<body>\n{body}\n</body>\n</html>\n"
    )
    with open(out_base + ".html", "w", encoding="utf-8") as f:
        f.write(html_doc)

    msg = f"kept {int(keep.sum())}/{len(keep)} blocks -> {out_base}.html ({len(body)} chars)"

    md = to_markdown(body)
    if md is not None:
        with open(out_base + ".md", "w", encoding="utf-8") as f:
            f.write(md)
        msg += f" + {out_base}.md"
    else:
        msg += "   (Markdown skipped: pip install html2text)"
    print(msg)


if __name__ == "__main__":
    main()
