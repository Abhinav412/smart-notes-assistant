"""Sentence-embedding search over the saved notes (optional dependency).

Uses sentence-transformers' all-MiniLM-L6-v2 (free, CPU-friendly, ~90 MB
downloaded once at first run). If the package or model is unavailable, the
app degrades to the TF-IDF search implemented in app/search.py — see
server.py / eval_search.py.
"""

from __future__ import annotations

MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingSearch:
    """Cosine similarity between sentence embeddings."""

    # MiniLM cosine scores for unrelated sentences are typically < 0.2.
    RELEVANT_CUTOFF = 0.30

    def __init__(self):
        self._model = None
        self._matrix = None
        self.notes: list[dict] = []

    @property
    def relevant_cutoff(self) -> float:
        return self.RELEVANT_CUTOFF

    @property
    def available(self) -> bool:
        try:
            return self._get_model() is not None
        except Exception:
            return False

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    def fit(self, notes: list[dict]) -> "EmbeddingSearch":
        self.notes = notes
        if not notes:
            self._matrix = None
            return self
        model = self._get_model()
        self._matrix = model.encode([n["text"] for n in notes], normalize_embeddings=True)
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self._matrix is None or not self.notes:
            return []
        model = self._get_model()
        qv = model.encode([query], normalize_embeddings=True)[0]
        sims = self._matrix @ qv
        order = sorted(range(len(sims)), key=lambda i: (-sims[i], self.notes[i]["id"]))
        return [(self.notes[i]["id"], float(sims[i])) for i in order][:k]
