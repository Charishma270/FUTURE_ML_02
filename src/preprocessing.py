"""Text preprocessing utilities for support ticket NLP models."""

from __future__ import annotations

import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize


FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}


def download_nltk_resources() -> None:
    """Download the NLTK resources required by the preprocessing pipeline."""
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for package, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass


@lru_cache(maxsize=1)
def _wordnet_available() -> bool:
    try:
        nltk.data.find("corpora/wordnet")
        return True
    except LookupError:
        return False


@lru_cache(maxsize=1)
def _stop_words() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return FALLBACK_STOPWORDS


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


def clean_text(text: object) -> str:
    """Clean raw support ticket text for TF-IDF based classification.

    Steps:
    - lowercase text
    - remove punctuation, digits, and special characters
    - tokenize
    - remove stopwords
    - lemmatize tokens
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = text.replace("{product_purchased}", "product")
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if _wordnet_available():
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = wordpunct_tokenize(text)
    else:
        tokens = wordpunct_tokenize(text)
    stop_words = _stop_words()
    lemmatizer = _lemmatizer()

    cleaned_tokens = []
    for token in tokens:
        if token in stop_words or len(token) <= 2:
            continue
        if _wordnet_available():
            cleaned_tokens.append(lemmatizer.lemmatize(token))
        else:
            cleaned_tokens.append(token)
    return " ".join(cleaned_tokens)


def add_cleaned_ticket_column(df, text_column: str = "ticket_text"):
    """Return a copy of df with a cleaned_ticket column."""
    df = df.copy()
    df["cleaned_ticket"] = df[text_column].apply(clean_text)
    return df
