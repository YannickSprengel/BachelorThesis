import justext


def extract_paragraph_features(html: str) -> list[dict]:
    """
    Use JusText for segmentation and feature extraction ONLY.
    """
    paragraphs = justext.justext(
        html,
        justext.get_stoplist("English"),
        length_low=0,  # don't filter out short paragraphs
        length_high=1000,
        stopwords_low=0.0,  # don't filter by stopword density
        stopwords_high=1.0,
        max_link_density=1.0  # don't filter by link density
    )

    features = []
    total = len(paragraphs)

    for i, para in enumerate(paragraphs):
        features.append({
            # Text content — this goes into tokenizer
            'text': para.text,

            # Structural metadata — these become structural features
            'dom_path': para.dom_path,  # e.g. "html/body/div/p"
            'xpath': para.xpath,
            'links_density': para.links_density,  # ratio of link text
            'stopwords_density': para.stopwords_density,
            'word_count': len(para.text.split()),
            'tag_name': para.tag_name,  # p, div, td, etc.

            # Position features
            'relative_position': i / (total + 1e-6),
            'is_heading': para.heading,  # was it in an h1-h6?

            'justext_label': para.class_type,  # 'good', 'bad', 'near-good'
        })

    return features