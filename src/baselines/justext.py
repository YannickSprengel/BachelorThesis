import justext

def run_justext_baseline(html: str, language: str = "English") -> str:
    paragraphs = justext.justext(html, justext.get_stoplist(language))
    return " ".join(p.text for p in paragraphs if not p.is_boilerplate)