"""Two search approaches, compared and reusable by app and evaluation.

- KeywordSearch : counts token overlap between the query and each note
  (simple baseline).
- TfidfSearch   : TF-IDF cosine similarity between the query and the saved
  notes (mirrors the classifier's features).

Both return (note_id, score) ranked lists. A result with score below
`relevant_cutoff` is marked `relevant: False`; the UI/API surface this as
low relevance instead of presenting a confident match.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer

from app.textnorm import QUERY_STOPWORDS, custom_stopwords


class KeywordSearch:
    """Baseline: fraction of query keywords that appear in the note text,
    plus a bonus when the whole query appears verbatim."""

    def __init__(self):
        self.notes: list[dict] = []
        self._tokenized: dict[str, set[str]] = {}

    def fit(self, notes: list[dict]) -> "KeywordSearch":
        self.notes = notes
        self._tokenized = {n["id"]: set(tokens(n["text"])) for n in notes}
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        qtokens = set(tokens(query)) - QUERY_STOPWORDS
        if not qtokens or not self.notes:
            return []
        low = query.lower()
        scored = []
        for note in self.notes:
            overlap = len(qtokens & self._tokenized.get(note["id"], set()))
            phrase = low.strip() in note["text"].lower()
            score = overlap / len(qtokens) + (0.5 if phrase else 0.0)
            if score > 0:
                scored.append((note["id"], score, phrase))
        phrase_first = sorted(scored, key=lambda s: (-int(s[2]), -s[1], s[0]))
        return [(nid, sc) for nid, sc, _ in phrase_first[:k]]


class TfidfSearch:
    """TF-IDF cosine similarity search over the saved notes."""

    RELEVANT_CUTOFF = 0.12

    def __init__(self):
        self.notes: list[dict] = []
        self._ids: list[str] = []
        self._matrix = None
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, stop_words=custom_stopwords
        )

    @property
    def relevant_cutoff(self) -> float:
        return self.RELEVANT_CUTOFF

    def fit(self, notes: list[dict]) -> "TfidfSearch":
        self.notes = notes
        self._ids = [n["id"] for n in notes]
        texts = [n["text"] for n in notes]
        if texts:
            self._matrix = self._vectorizer.fit_transform(texts)
        else:
            self._matrix = None
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self._matrix is None or not self._ids:
            return []
        qv = self._vectorizer.transform([query])
        sims = (self._matrix @ qv.T).toarray().ravel()
        order = sorted(range(len(sims)), key=lambda i: (-sims[i], self._ids[i]))
        return [
            (self._ids[i], float(sims[i])) for i in order if sims[i] > 0.0
        ][:k]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9₹]+", text.lower())
