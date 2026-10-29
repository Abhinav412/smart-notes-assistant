"""Shared text normalisation helpers (training + search use the same list)."""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# sklearn's built-in 'english' stop list (Glasgow IR list) also removes words
# that carry category signal for this dataset, e.g. "bill" (Money), "due"
# (Money), "call"/"put" (Travel/chores). We keep the generic list but exclude
# those domain-relevant terms. Chosen via experiments.py (CV on train only).
DOMAIN_RELEVANT_WORDS = {
    "bill", "due", "call", "put", "system", "name", "become", "fill",
    "found", "whole", "cry",
}

custom_stopwords = sorted(set(ENGLISH_STOP_WORDS) - DOMAIN_RELEVANT_WORDS)

# Tokens ignored by the keyword search baseline (question phrasing noise).
QUERY_STOPWORDS = {
    "a", "am", "an", "and", "any", "are", "at", "by", "do", "does", "for",
    "i", "in", "is", "it", "me", "my", "need", "of", "on", "or", "the",
    "this", "to", "want", "what", "when", "where", "which", "who", "whom",
    "why", "you",
}

TOKEN_RE = re.compile(r"[a-z0-9₹]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alnum/rupee tokens (keeps numbers such as '2' '000')."""
    return TOKEN_RE.findall(text.lower())
