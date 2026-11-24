"""FastAPI backend: POST /classify (API contract) + note management and search.

Start with:
    uvicorn app.server:app --port 8000
(see README.md). The UI is served at http://127.0.0.1:8000/
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.classifier import load_classifier
from app.notes_store import NotesStore
from app.search import KeywordSearch, TfidfSearch

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ["Study", "Money", "Travel", "Shopping"]


class TextIn(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(
                "text must be a JSON string with at least one "
                "non-whitespace character"
            )
        return v  # keep raw text for storage; model gets it as-is


class NoteServiceState:
    """Model, store and search engines, loaded once at startup."""

    def __init__(self):
        self.classifier = load_classifier()
        self.store = NotesStore()
        from app.embeddings import EmbeddingSearch  # optional dependency

        try:
            emb = EmbeddingSearch()
            emb.fit(self.store.all_notes())
            self.search_mode = "embeddings"
            self.search_index = emb
        except Exception:
            self.search_mode = "tfidf"
            self.search_index = TfidfSearch().fit(self.store.all_notes())
        self.keyword_index = KeywordSearch().fit(self.store.all_notes())

    def refit_search(self) -> None:
        self.search_index.fit(self.store.all_notes())
        self.keyword_index.fit(self.store.all_notes())

    def classify_pending(self) -> int:
        """Give model predictions to notes saved without a category yet."""
        n = 0
        if self.store is None:
            return 0
        for note in self.store.all_notes():
            if note.get("category") == "":
                result = self.classifier.classify(note["text"])
                note["category"] = result["category"]
                note["uncertain"] = result["uncertain"]
                n += 1
        if n:
            self.store.save()
        return n


STATE: NoteServiceState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global STATE
    STATE = NoteServiceState()
    # Smoke-test the model once at startup. If the environment does not
    # match the one that trained the bundle, predict_proba can raise at
    # request time (seen as a bare HTTP 500 on the first "Add note").
    # Failing here makes the cause visible at boot instead.
    STATE.classifier.classify("startup smoke test")
    STATE.classify_pending()
    # Pre-load the embedding model once so the first search/first note-add
    # (and the CLI's load-starter call) never trigger a slow lazy load.
    try:
        if hasattr(STATE.search_index, "_model") and STATE.search_index._model is None:
            STATE.search_index._get_model()
    except Exception:
        pass  # fallback search will be used instead
    yield


app = FastAPI(title="Smart Notes Assistant", lifespan=lifespan)


def require_state() -> NoteServiceState:
    if STATE is None:  # pragma: no cover - only in tests without lifespan
        raise HTTPException(status_code=503, detail="server is starting up")
    return STATE


# --- API contract: POST /classify -------------------------------------------
@app.post("/classify")
def classify(body: TextIn):
    result = require_state().classifier.classify(body.text)
    return {
        "category": result["category"],
        "uncertain": result["uncertain"],
        "confidence": round(result["confidence"], 4),
        "margin": round(result["margin"], 4),
    }


# --- Notes API --------------------------------------------------------------
@app.post("/api/notes", status_code=201)
def add_note(body: TextIn):
    state = require_state()
    result = state.classifier.classify(body.text)
    note = state.store.add(
        body.text, result["category"], result["uncertain"]
    )
    state.refit_search()
    return note


@app.delete("/api/notes/{note_id}", status_code=204)
def delete_note(note_id: str):
    state = require_state()
    if not state.store.delete(note_id):
        raise HTTPException(
            status_code=404, detail=f"note {note_id!r} not found"
        )
    # The search indices hold in-memory state over the notes, so refit them
    # right away — otherwise the deleted note would keep showing up in
    # search results until the server restarts.
    state.refit_search()
    return Response(status_code=204)


@app.get("/api/notes")
def list_notes(
    category: Optional[str] = Query(default=None),
    uncertain: Optional[bool] = Query(default=None),
):
    state = require_state()
    if category is not None and category not in CATEGORIES:
        raise HTTPException(status_code=422, detail=f"unknown category {category!r}")
    notes = state.store.filtered(category=category, uncertain=uncertain)
    state.classify_pending()
    return {"notes": notes}


@app.post("/api/notes/load-starter")
def load_starter():
    state = require_state()
    summary = state.store.load_starter()
    state.classify_pending()
    state.refit_search()
    return summary


@app.get("/api/search")
def search(q: str = Query(..., min_length=1)):
    state = require_state()
    q = q.strip()
    if not q:
        raise HTTPException(status_code=422, detail="empty query")
    ranked = state.search_index.search(q, k=10)
    cutoff = state.search_index.relevant_cutoff
    results = []
    for note_id, score in ranked:
        note = state.store.get(note_id)
        if note is None:
            continue
        results.append(
            {
                "id": note["id"],
                "text": note["text"],
                "category": note.get("category"),
                "uncertain": note.get("uncertain"),
                "score": round(score, 4),
                "relevant": score >= cutoff,
            }
        )
    message = None
    if not results or not any(r["relevant"] for r in results):
        message = "No good match found for this query."
    return {
        "query": q,
        "search_mode": state.search_mode,
        "relevant_cutoff": cutoff,
        "results": results,
        "message": message,
    }


@app.get("/api/model-info")
def model_info():
    return {"model": require_state().classifier.description}


# Unhandled errors: return the exception text inside `detail` so the UI's
# error box (it already displays r.body.detail) can show the real cause
# instead of a bare "HTTP 500". Starlette still logs the full traceback.
@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"internal error: {exc}"},
    )


# --- Static UI --------------------------------------------------------------
STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
