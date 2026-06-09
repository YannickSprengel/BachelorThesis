"""
Pipeline overview:
    1. DripperPreprocessor  – wraps simplify_html(), returns (simplified_blocks, mapping_blocks)
    2. BlockFeatureExtractor – turns each BeautifulSoup block into a fixed-dim numpy vector
    3. LabelGenerator        – assigns binary labels (1=content, 0=boilerplate) per block
    4. HTMLExtractionDataset – PyTorch Dataset: one sample = one document
    5. collate_fn            – pads variable-length sequences for batching
"""

import re
import json
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from bs4 import BeautifulSoup, Tag


# Wraps mineru_html's simplify_html to give two parallel block sequences.
class DripperPreprocessor:
    def __init__(self):
        from mineru_html.process.simplify_html import simplify_html
        self._simplify_fn = simplify_html

    def process(self, raw_html: str) -> tuple[list[Tag], list[Tag]]:
        simplified_html_str, mapping_html_str = self._simplify_fn(raw_html)
        simplified_blocks = self._parse_blocks(simplified_html_str)
        mapping_blocks    = self._parse_blocks(mapping_html_str)

        return simplified_blocks, mapping_blocks

    @staticmethod
    def _parse_blocks(html_str: str) -> list[Tag]:
        """
        Parse the simplified/mapping HTML string and return all top-level
        blocks (elements with an _item_id attribute) as a flat ordered list.
        """
        soup = BeautifulSoup(html_str, 'html.parser')
        blocks = soup.find_all(attrs={'_item_id': True})
        # Sort by _item_id to guarantee order matches between both branches
        blocks.sort(key=lambda b: int(b.get('_item_id', 0)))
        return blocks

    @staticmethod
    def reconstruct_content(mapping_blocks: list[Tag],
                            labels: list[int]) -> str:
        """
        Post-processing: apply predicted labels to mapping blocks to reconstruct
        the main content as HTML. This is Stage 3 of the Dripper pipeline.

        Args:
            mapping_blocks: list of BS4 Tags from the Mapping HTML branch
            labels:         predicted binary labels (1=content, 0=boilerplate)

        Returns:
            HTML string of the extracted main content
        """
        assert len(mapping_blocks) == len(labels)
        selected = [str(block) for block, label in zip(mapping_blocks, labels) if label == 1]
        return "\n".join(selected)

# STEP 2 – BLOCK FEATURE EXTRACTOR
# Converts a single BeautifulSoup block Tag → a fixed-length numpy float32 vector

# All HTML tags we recognise (anything else → 'other' slot)
_TAG_VOCAB = [
    'div', 'p', 'span', 'a',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'li', 'ul', 'ol',
    'table', 'td', 'th', 'tr',
    'article', 'section', 'main', 'aside',
    'blockquote', 'pre', 'code',
    'figure', 'figcaption', 'img',
    'other'   # catch-all
]

_CONTENT_KEYWORDS = [
    'article', 'content', 'main', 'post', 'body', 'text',
    'story', 'blog', 'entry', 'detail', 'description'
]
_BOILERPLATE_KW = [
    'nav', 'menu', 'footer', 'header', 'sidebar', 'ad', 'ads',
    'banner', 'social', 'share', 'comment', 'related',
    'recommend', 'cookie', 'popup', 'breadcrumb', 'pagination'
]

# FEATURE_DIM is the total number of features per block.
FEATURE_DIM = (
    len(_TAG_VOCAB)       # 28  tag one-hot
    + 8                   #  8  text statistics
    + 2                   #  2  link features
    + 2                   #  2  nesting depth + child tag count
    + 1                   #  1  relative document position
    + 2                   #  2  content / boilerplate keyword scores
    + 4                   #  4  binary child-tag flags (table, code, img, heading)
)

class BlockFeatureExtractor:
    """
    Extracts a fixed-length (FEATURE_DIM = 47) float32 feature vector from a
    single BeautifulSoup block Tag produced by DripperPreprocessor.

    Feature groups:
        [0:28]   – one-hot tag type
        [28:36]  – text statistics (word count, char count, punct density, ...)
        [36:38]  – link density, link count
        [38:40]  – DOM depth, child element count
        [40]     – relative position in document (0.0 → 1.0)
        [41:43]  – content keyword score, boilerplate keyword score
        [43:47]  – binary flags: has_table, has_code, has_img, has_heading
    """

    def extract(self, block: Tag, block_idx: int, total_blocks: int) -> np.ndarray:
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        ptr = 0

        # ── Group 1: Tag type one-hot (28 dims) ──────────────────────────────
        tag_name = (block.name or 'other').lower()
        tag_idx = _TAG_VOCAB.index(tag_name) if tag_name in _TAG_VOCAB else _TAG_VOCAB.index('other')
        features[ptr + tag_idx] = 1.0
        ptr += len(_TAG_VOCAB)  # ptr = 28

        # ── Group 2: Text statistics (8 dims) ────────────────────────────────
        text = block.get_text(separator=' ', strip=True)
        words = text.split()
        word_count  = len(words)
        char_count  = len(text)

        features[ptr + 0] = min(char_count  / 1000.0, 1.0)          # normalised char count
        features[ptr + 1] = min(word_count  / 200.0,  1.0)          # normalised word count
        features[ptr + 2] = len(re.findall(r'[.!?]', text)) / max(word_count, 1)   # sent-end punct density
        features[ptr + 3] = sum(1 for c in text if c.isupper()) / max(char_count, 1)  # upper ratio
        features[ptr + 4] = np.mean([len(w) for w in words]) / 10.0 if words else 0.0  # avg word len
        features[ptr + 5] = len(re.findall(r'\d', text)) / max(char_count, 1)     # digit density
        features[ptr + 6] = len(re.findall(r'[,;:]', text)) / max(word_count, 1)  # comma/semicol density
        features[ptr + 7] = min(len(re.findall(r'\n', text)) / 10.0, 1.0)         # newline count
        ptr += 8  # ptr = 36

        # ── Group 3: Link features (2 dims) ──────────────────────────────────
        anchors = block.find_all('a')
        anchor_text_len = sum(len(a.get_text()) for a in anchors)
        features[ptr + 0] = anchor_text_len / max(char_count, 1)    # link density
        features[ptr + 1] = min(len(anchors) / 10.0, 1.0)           # normalised link count
        ptr += 2  # ptr = 38

        # ── Group 4: DOM nesting (2 dims) ─────────────────────────────────────
        all_descendants = block.find_all(True)
        features[ptr + 0] = min(len(all_descendants) / 30.0, 1.0)   # child element count
        features[ptr + 1] = min(self._max_depth(block) / 8.0, 1.0)  # max nesting depth
        ptr += 2  # ptr = 40

        # ── Group 5: Document position (1 dim) ────────────────────────────────
        features[ptr] = block_idx / max(total_blocks - 1, 1)         # 0.0 → 1.0
        ptr += 1  # ptr = 41

        # ── Group 6: Class / ID keyword scores (2 dims) ────────────────────────
        class_id_str = ' '.join([
            ' '.join(block.get('class', [])),
            block.get('id', '')
        ]).lower()
        content_hits     = sum(kw in class_id_str for kw in _CONTENT_KEYWORDS)
        boilerplate_hits = sum(kw in class_id_str for kw in _BOILERPLATE_KW)
        features[ptr + 0] = min(content_hits    / 3.0, 1.0)
        features[ptr + 1] = min(boilerplate_hits / 3.0, 1.0)
        ptr += 2  # ptr = 43

        #Group 7: Binary child-tag flags (4 dims)
        features[ptr + 0] = 1.0 if block.find('table') else 0.0
        features[ptr + 1] = 1.0 if block.find(['code', 'pre']) else 0.0
        features[ptr + 2] = 1.0 if block.find('img') else 0.0
        features[ptr + 3] = 1.0 if block.find(['h1','h2','h3','h4','h5','h6']) else 0.0

        return features

    @staticmethod
    def _max_depth(tag: Tag, depth: int = 0) -> int:
        """Recursively compute max nesting depth of a BS4 tag."""
        children = [c for c in tag.children if isinstance(c, Tag)]
        if not children:
            return depth
        return max(BlockFeatureExtractor._max_depth(c, depth + 1) for c in children)


# Given a document's blocks and ground-truth annotation, assign binary labels.
class LabelGenerator:
    """
    Assigns a binary label (1=content, 0=boilerplate) to each simplified block.

    Supports three ground-truth formats:

    1. WebMainBench format:  raw HTML contains cc-select=True attributes
       → check if any descendant has cc-select="True"

    2. main_html string format: ground-truth is a separate HTML string
       → check text overlap between block text and main content text

    3. Dripper silver labels: run Dripper-0.6B to generate pseudo-labels
       → useful for training on unlabelled CommonCrawl data
    """

    def __init__(self, overlap_threshold: float = 0.5):
        self.overlap_threshold = overlap_threshold

    def from_cc_select_attrs(self, simplified_blocks: list[Tag]) -> list[int]:
        labels = []
        for block in simplified_blocks:
            selected = (
                block.get('cc-select') == 'True'
                or bool(block.find(attrs={'cc-select': 'True'}))
            )
            labels.append(1 if selected else 0)
        return labels

    def from_main_html(self,
                       simplified_blocks: list[Tag],
                       main_html: str) -> list[int]:
        """
        Format 2 (generic ground truth): labels blocks by comparing their
        text against the ground-truth main content HTML string.

        Strategy: for each block, compute what fraction of its words appear
        in the main content text. Label as content if fraction ≥ threshold.
        """
        main_text = BeautifulSoup(main_html, 'html.parser').get_text().lower()
        main_words = set(main_text.split())

        labels = []
        for block in simplified_blocks:
            block_words = block.get_text().lower().split()
            if not block_words:
                labels.append(0)
                continue
            overlap = sum(1 for w in block_words if w in main_words) / len(block_words)
            labels.append(1 if overlap >= self.overlap_threshold else 0)
        return labels

# One sample = one HTML document. Returns (feature_matrix, label_vector).
class HTMLExtractionDataset(Dataset):
    """
    Each item is a document represented as:
        features: FloatTensor (seq_len, FEATURE_DIM)  — block feature matrix
        labels:   LongTensor  (seq_len,)               — binary labels per block
        item_ids: list[int]                            — block IDs (for reconstruction)

    Usage:
        dataset = HTMLExtractionDataset.from_webmainbench("path/to/bench.jsonl")
        loader  = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)
        for features, labels, lengths in loader:
            ...  # feed into your BiLSTM
    """

    def __init__(self,
                 preprocessor: DripperPreprocessor,
                 feature_extractor: BlockFeatureExtractor,
                 label_generator: LabelGenerator):
        self.preprocessor       = preprocessor
        self.feature_extractor  = feature_extractor
        self.label_generator    = label_generator
        self._samples: list[dict] = []   # stores processed documents

    # ── Loading helpers ────────────────────────────────────────────────────

    @classmethod
    def from_webmainbench(cls,
                          jsonl_path: str,
                          max_docs: Optional[int] = None) -> 'HTMLExtractionDataset':
        ds = cls(DripperPreprocessor(), BlockFeatureExtractor(), LabelGenerator())
        path = Path(jsonl_path)

        with path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_docs and i >= max_docs:
                    break
                sample = json.loads(line.strip())
                ds.add_document(
                    raw_html=sample['html'],
                    main_html=sample.get('main_html'),
                    label_format='cc_select',
                    meta=sample.get('meta', {})
                )
        print(f"[Dataset] Loaded {len(ds)} documents from {path.name}")
        return ds

    @classmethod
    def from_html_directory(cls,
                            html_dir: str,
                            label_format: str = 'main_html') -> 'HTMLExtractionDataset':
        ds = cls(DripperPreprocessor(), BlockFeatureExtractor(), LabelGenerator())
        html_dir = Path(html_dir)

        for html_file in html_dir.glob('*.html'):
            txt_file = html_file.with_suffix('.txt')
            if not txt_file.exists():
                continue
            raw_html  = html_file.read_text(encoding='utf-8', errors='ignore')
            main_text = txt_file.read_text(encoding='utf-8', errors='ignore')
            ds.add_document(raw_html=raw_html, main_html=main_text, label_format='main_html')

        print(f"[Dataset] Loaded {len(ds)} documents from {html_dir}")
        return ds

    def add_document(self,
                     raw_html: str,
                     main_html: Optional[str] = None,
                     label_format: str = 'cc_select',
                     meta: dict = None):
        try:
            simplified_blocks, mapping_blocks = self.preprocessor.process(raw_html)
            # Also store the raw HTML strings for Stage 3 reconstruction via map_to_main
            simp_html_str, map_html_str = self.preprocessor._simplify_fn(raw_html)
        except Exception as e:
            print(f"[Dataset] Skipping document: preprocessing failed: {e}")
            return

        if len(simplified_blocks) == 0:
            return

        # Feature extraction: one vector per block
        feature_matrix = np.stack([
            self.feature_extractor.extract(block, idx, len(simplified_blocks))
            for idx, block in enumerate(simplified_blocks)
        ])  # shape: (seq_len, FEATURE_DIM)

        # Label generation
        if label_format == 'cc_select':
            labels = self.label_generator.from_cc_select_attrs(simplified_blocks)
        elif label_format == 'main_html' and main_html:
            labels = self.label_generator.from_main_html(simplified_blocks, main_html)
        elif label_format == 'unlabelled':
            labels = [-1] * len(simplified_blocks)  # -1 = unknown, for inference
        else:
            raise ValueError(f"Unknown label_format='{label_format}' or missing main_html.")

        # Store item_ids so we can reconstruct content after prediction
        item_ids = [int(b.get('_item_id', i)) for i, b in enumerate(simplified_blocks)]

        self._samples.append({
            'features':            feature_matrix,
            'labels':              np.array(labels, dtype=np.int64),
            'item_ids':            item_ids,
            'simplified_blocks':   simplified_blocks,    # list[Tag] for label_dict building
            'mapping_blocks':      mapping_blocks,        # list[Tag] (legacy fallback)
            'simplified_html_str': simp_html_str,         # str for map_to_main
            'mapping_html_str':    map_html_str,          # str for map_to_main
            'seq_len':             len(simplified_blocks),
            'meta':                meta or {},
        })

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._samples[idx]
        features = torch.tensor(sample['features'], dtype=torch.float32)
        labels   = torch.tensor(sample['labels'],   dtype=torch.long)
        return features, labels

    def get_raw_sample(self, idx: int) -> dict:
        """Return the full sample dict (including mapping_blocks for reconstruction)."""
        return self._samples[idx]

def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """
    Pads a batch of variable-length documents to the length of the longest
    document in the batch.
    """
    feature_list, label_list = zip(*batch)

    lengths = torch.tensor([f.size(0) for f in feature_list], dtype=torch.long)

    features_padded = pad_sequence(feature_list, batch_first=True, padding_value=0.0)
    labels_padded   = pad_sequence(label_list,   batch_first=True, padding_value=-100)

    max_len = features_padded.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return features_padded, labels_padded, lengths, mask


class FeatureNormalizer:
    """
    Fits a StandardScaler on training data feature vectors.
    Should be fit only on the training split, then applied to val/test.
    """

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, dataset: HTMLExtractionDataset):
        # Stack all feature vectors from all documents
        all_features = np.vstack([s['features'] for s in dataset._samples])
        self.scaler.fit(all_features)
        self._fitted = True
        print(f"[Normalizer] Fitted on {len(all_features)} block feature vectors.")

    def transform(self, dataset: HTMLExtractionDataset) -> HTMLExtractionDataset:
        assert self._fitted, "Call .fit() before .transform()"
        for sample in dataset._samples:
            sample['features'] = self.scaler.transform(sample['features']).astype(np.float32)
        return dataset

    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        self._fitted = True


def _load_process_fns():
    """
    Load Stage 3 functions from mineru_html.process.
    Returns (map_to_main_fn, convert2content_fn).
    """
    try:
        from mineru_html.process.map_to_main import map_to_main
        from mineru_html.process.convert2content import convert2content
        from mineru_html.process.parse_result import parse_result
        print("[Process] Loaded map_to_main, convert2content, parse_result from mineru_html.process")
        return map_to_main, convert2content, parse_result
    except ImportError:
        print("[Process] WARNING: mineru_html.process not available. "
              "Using fallback reconstruction.")
        return _fallback_map_to_main, _fallback_convert2content, _fallback_parse_result


def _fallback_map_to_main(mapping_html: str, labels: dict[str, str]) -> str:
    """
    Fallback: select blocks from mapping_html whose _item_id maps to 'main'.
    """
    soup = BeautifulSoup(mapping_html, 'html.parser')
    blocks = soup.find_all(attrs={'_item_id': True})
    selected = [str(b) for b in blocks if labels.get(b['_item_id']) == 'main']
    return "\n".join(selected)


def _fallback_convert2content(main_html: str, output_format: str = 'txt') -> str:
    """Fallback: strip tags and return plain text."""
    return BeautifulSoup(main_html, 'html.parser').get_text(separator='\n', strip=True)


def _fallback_parse_result(raw_output: str) -> dict[str, str]:
    """Fallback: parse the JSON/compact label output from a model."""
    try:
        return json.loads(raw_output)
    except Exception:
        return {}


class ContentReconstructor:
    """
    Stage 3 of the Dripper pipeline: applies your LSTM's predicted labels
    back to the Mapping HTML to produce clean extracted content.
    """

    def __init__(self):
        self.map_to_main, self.convert2content, self.parse_result = _load_process_fns()

    def labels_to_dripper_format(self,
                                  simplified_blocks: list[Tag],
                                  predicted_labels: list[int]) -> dict[str, str]:
        label_dict = {}
        for block, label in zip(simplified_blocks, predicted_labels):
            item_id = block.get('_item_id', '0')
            label_dict[str(item_id)] = 'main' if label == 1 else 'other'
        return label_dict

    def reconstruct(self,
                    dataset: HTMLExtractionDataset,
                    doc_idx: int,
                    predicted_labels: list[int],
                    output_format: str = 'txt') -> str:
        """
        Full Stage 3: predicted labels → clean extracted content.
        """
        sample = dataset.get_raw_sample(doc_idx)
        mapping_blocks = sample['mapping_blocks']
        mapping_html_str = sample.get('mapping_html_str', '')

        # Build the label dict in Dripper's format
        simp_blocks = dataset.preprocessor._parse_blocks(
            sample.get('simplified_html_str', '')
        ) if sample.get('simplified_html_str') else mapping_blocks

        label_dict = self.labels_to_dripper_format(simp_blocks, predicted_labels)

        # Stage 3a: map labels → select content blocks from Mapping HTML
        main_html = self.map_to_main(mapping_html_str, label_dict)

        # Stage 3b: convert HTML → desired output format
        return self.convert2content(main_html, output_format)