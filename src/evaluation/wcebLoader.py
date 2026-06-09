"""
Pages are matched between the two via page_id
"""
import argparse
import os
import glob
import json


def read_wceb(combined_dir, datasets=None):
    """
    Yield (dataset, page_id, html, plaintext) for every page in the WCEB combined set.
    """
    truth_dir = os.path.join(combined_dir, "ground-truth")
    html_dir = os.path.join(combined_dir, "html")

    if datasets is None:
        datasets = sorted(os.path.splitext(f)[0]
                          for f in os.listdir(truth_dir) if f.endswith(".jsonl"))

    for ds in datasets:
        truth = {}
        with open(os.path.join(truth_dir, f"{ds}.jsonl"), encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                truth[j["page_id"]] = j.get("plaintext", "")

        for path in glob.glob(os.path.join(html_dir, ds, "*.html")):
            page_id = os.path.splitext(os.path.basename(path))[0]
            if page_id not in truth:
                continue
            html = open(path, encoding="utf-8", errors="ignore").read()
            yield ds, page_id, html, truth[page_id]