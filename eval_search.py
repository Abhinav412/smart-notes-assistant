#!/usr/bin/env python3
"""Measure search quality (hit rate @3) for two approaches.

    python eval_search.py

Corpus: the 12 starter notes (data/starter_notes.json) — the same set the
brief uses. Queries: data/search_queries.json (16 queries, 6 share no words
with their target). Reports hit rate @3 for a keyword-overlap baseline and
for TF-IDF cosine similarity, plus the queries each approach fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.search import KeywordSearch, TfidfSearch, tokens

ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def hit_at_k(results: list[tuple[str, float]], expected: list[str], k: int = 3):
    top = [nid for nid, _ in results[:k]]
    return any(nid in top for nid in expected)


def main():
    global NOTES_BY_ID
    notes = json.loads((ROOT / "data" / "starter_notes.json").read_text(encoding="utf-8"))
    queries = json.loads((ROOT / "data" / "search_queries.json").read_text(encoding="utf-8"))
    NOTES_BY_ID = {n["id"]: n for n in notes}

    zero_overlap = sum(
        1
        for q in queries
        if not any(
            set(tokens(q["query"]))
            & (set(tokens(NOTES_BY_ID[e]["text"])) if e in NOTES_BY_ID else set())
            for e in q["expected"]
        )
    )
    print(f"Corpus: {len(notes)} starter notes; queries: {len(queries)} "
          f"(no word overlap with target: {zero_overlap})\n")

    engines = {"keyword": KeywordSearch().fit(notes), "tfidf": TfidfSearch().fit(notes)}
    try:
        from app.embeddings import EmbeddingSearch

        emb = EmbeddingSearch()
        emb.fit(notes)
        engines["embeddings"] = emb
    except Exception as exc:  # optional dependency missing / offline
        print(f"(embeddings unavailable: {exc.__class__.__name__}: {exc})")
    scores = {name: [] for name in engines}

    print(f"{'query':<28}{'expected':<14}" + "".join(f"{name:<30}" for name in engines))
    for q in queries:
        row = f"{q['query'][:27]:<28}{','.join(q['expected']):<14}"
        for name, engine in engines.items():
            res = engine.search(q["query"], k=3)
            hit = hit_at_k(res, q["expected"])
            scores[name].append(hit)
            shown = " ".join(f"{nid}:{s:.2f}" for nid, s in res[:3])
            row += f"{'HIT ' if hit else 'MISS'} top3=({shown:<22})"
        print(row)

    print()
    for name, hits in scores.items():
        print(f"{name:<8} hit rate @3 = {sum(hits)}/{len(hits)} = {sum(hits)/len(hits):.2%}")

    print("\nNotes on low-relevance handling:")
    for phrase in ["xyzzy plugh", "quantum velcro flood"]:
        res = engines["tfidf"].search(phrase, k=3)
        over = [
            (nid, s) for nid, s in res
            if s >= engines["tfidf"].relevant_cutoff
        ]
        raw = " ".join(f"{nid}:{s:.2f}" for nid, s in res[:3])
        print(f"  {phrase!r}: raw=({raw or 'none'}) -> relevant results: {over or 'none (below cutoff)'}")


if __name__ == "__main__":
    main()
