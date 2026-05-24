import re
import spacy

_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

_EMAIL = re.compile(r"\S+@\S+")
_PHONE = re.compile(r"\+?\d[\d\-\s()]{7,}\d")
_URL = re.compile(r"https?://\S+|www\.\S+")
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = _EMAIL.sub(" ", text)
    text = _PHONE.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _WS.sub(" ", text)
    return text.strip()


def normalize_tokens(text: str) -> list[str]:
    """Lowercased lemmas, no stopwords/punctuation — for skill matching."""
    doc = _nlp(text.lower())
    return [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and t.text.strip()]


def preprocess(text: str) -> dict:
    cleaned = clean_text(text)
    return {"clean": cleaned, "tokens": normalize_tokens(cleaned)}