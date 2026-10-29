"""Search-quality unit tests for both engines."""

import json
from pathlib import Path

import pytest

from app.search import KeywordSearch, TfidfSearch, tokens
from app.textnorm import QUERY_STOPWORDS

ROOT = Path(__file__).resolve().parent.parent
STARTER = json.loads(
    (ROOT / "data" / "starter_notes.json").read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def engines():
    return {
        "keyword": KeywordSearch().fit(STARTER),
        "tfidf": TfidfSearch().fit(STARTER),
    }


def test_find_by_word(engines):
    for engine in engines.values():
        hits = [nid for nid, _ in engine.search("hotel reservation", k=3)]
        assert hits[0] == "N007"


def test_zero_overlap_synonym_queries_fail_without_embeddings(engines):
    # Expected limitation of bag-of-words methods (see RESULTS.md):
    # "money transfer" shares no token with the savings note.
    for engine in engines.values():
        hits = [nid for nid, _ in engine.search("money transfer", k=3)]
        assert "N006" not in hits


def test_nonsense_query_has_no_relevant_result(engines):
    for engine in engines.values():
        results = engine.search("xyzzy plugh", k=3)
        assert all(score < engine.relevant_cutoff for _, score in results)


def test_query_stopwords_are_dropped():
    assert "when" not in (set(tokens("when is my flight?")) - QUERY_STOPWORDS)
