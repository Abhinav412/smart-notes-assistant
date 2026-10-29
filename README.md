# Smart Notes Assistant

A small local app that saves personal notes, classifies them into `Study`,
`Money`, `Travel`, `Shopping` with a **trained scikit-learn model**, marks
notes the model is **unsure** about, and **searches** saved notes — including
with phrase queries that share no words with the matching note.

Everything runs locally on CPU. No paid APIs, keys, or cloud deployment.

## What's inside

| Path | Purpose |
| --- | --- |
| `app/server.py` | FastAPI backend: `POST /classify` (API contract), notes CRUD, search, serves the UI |
| `app/static/index.html` | Browser UI (add, filter, search, Uncertain badges) |
| `app/classifier.py` | Loads the model bundle once, applies the uncertainty rule |
| `app/notes_store.py` | JSON persistence, starter-note loading, ID allocation |
| `app/search.py` | Keyword-overlap baseline and TF-IDF cosine search |
| `app/embeddings.py` | Sentence-embedding search (all-MiniLM-L6-v2), optional with fallback |
| `train.py` | One command: trains baselines + final model, prints metrics, saves the bundle |
| `experiments.py` | Model/config selection via 5-fold CV on **training data only** |
| `threshold_analysis.py` | Uncertainty signal comparison (validation + out-of-scope) |
| `eval_search.py` | Search evaluation: hit rate @3 per approach |
| `error_evidence.py` | Feature-level evidence for the mistakes analysed in `RESULTS.md` |
| `data/` | Supplied training/validation/OOS data, starter notes, search test set |
| `tests/` | `pytest` suite (API contract, persistence, search) |
| `RESULTS.md` | All measurements: model metrics, errors, search, uncertainty trade-offs |
| `AI_WORKFLOW.md` | How AI tools were used, and the bugs caught |

## Setup (once)

Python 3.10+ recommended.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

The optional `sentence-transformers` package downloads the
`all-MiniLM-L6-v2` embedding model (~90 MB) from the Hugging Face hub on
first use — no account or key needed. **If this package or the download is
unavailable the app automatically falls back to the built-in TF-IDF search**
(hit rate @3 0.56 vs 0.94 — similarity of synonyms is what changes).

## Train the model (one command)

```bash
python train.py
```

This fits both baselines and the final TF-IDF + LogisticRegression pipeline on
`data/train.csv` **only**, prints accuracy and macro-F1 on
`data/validation.csv`, sweeps the uncertainty threshold, and saves
`models/classifier.joblib` (which the app and `/classify` load once at
startup). Expected output: trained model ≈ **0.86 accuracy / 0.86 macro-F1**.

## Run the app (one command; UI and /classify in the same server)

```bash
uvicorn app.server:app --port 8000
```

- UI: <http://127.0.0.1:8000/> — add notes, filter by category or "Uncertain",
  search, load starter notes by button.
- API: `POST /classify` per the contract. Example:

POSIX shell:

```bash
curl -X POST http://127.0.0.1:8000/classify -H "Content-Type: application/json" -d "{\"text\":\"Pay my electricity bill tomorrow.\"}"
# -> {"category":"Money","uncertain":false,"confidence":0.4138,"margin":0.1872}
```

Windows PowerShell (no curl needed):

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/classify -Method Post `
  -ContentType "application/json" `
  -Body (@{text = "Pay my electricity bill tomorrow."} | ConvertTo-Json)
```


Smoke test the contract with the provided helper:

```bash
python examples/test_api.py --url http://127.0.0.1:8000/classify
```

Notes are stored in `notes.json` (created on first save) and survive restarts.
Reset the store by deleting that file.

### Load the starter notes

Any one of these:

- Click **Load starter notes** in the UI,
- `python -m app.load_starter` (CLI), or
- `curl -X POST http://127.0.0.1:8000/api/notes/load-starter`

Starter notes keep their original IDs `N001`–`N012` and are classified by the
trained model on load. Loading again adds nothing new (idempotent). Notes you
add get IDs prefixed `U` and never reuse the reserved `N001`–`N012` range.

While the server is running it owns `notes.json`; the CLI detects this and
routes its request through the server automatically, so order never matters
(and a direct file write can never be clobbered by the server's next save).
With no server running, the CLI writes the store file directly.

> Note: the very first server start can take up to ~1 minute on a fresh
> machine while the embedding model is fetched/validated from the local
> Hugging Face cache (or hub). Subsequent starts are in seconds. Without
> `sentence-transformers` you get the instant TF-IDF fallback instead.

## Tests

```bash
python -m pytest
```

Covers: `/classify` contract (valid note, every invalid-input case from the
contract incl. malformed JSON, repeatability, latency budget via response
sanity), semantic sanity of familiar phrases, persistence round-trip,
idempotent starter loading, user-ID allocation, category filtering, and
search behaviour (including the anti-hallucination "no good match" path).
Tests run offline; `tests/test_classifier_api.py` skips if the model bundle
hasn't been trained yet.

## Reproduce every number in RESULTS.md

```bash
python train.py               # model/baseline metrics + threshold sweep
python experiments.py         # CV comparison of preprocessing choices (train-only)
python threshold_analysis.py  # uncertainty signal trade-off table
python eval_search.py         # search hit rate @3 per approach
python error_evidence.py      # feature-level evidence for analysed mistakes
```

## API summary

| Method & path | Purpose |
| --- | --- |
| `POST /classify` | `{"text": "..."}` → `{category, uncertain, confidence, margin}` (contract) |
| `POST /api/notes` | Add a note (classified + persisted), returns the stored note |
| `GET /api/notes?category=&uncertain=` | List / filter notes |
| `POST /api/notes/load-starter` | Idempotent starter-note load |
| `GET /api/search?q=` | Ranked results; `relevant` flag + explicit "No good match" message |

Invalid `/classify` input (blank/whitespace, missing `text`, non-string text,
malformed JSON) returns HTTP 422; the server keeps serving afterwards.

## Design decisions in brief

- **Single FastAPI server** serves UI and `/classify` — one process, no
  CORS/dual-startup confusion (the contract forbids Streamlit-only, so this
  avoids running two servers).
- **Uncertainty rule**: flag when the model's probability margin
  `P(top) − P(2nd class)` falls below a tuned cutoff — the best-performing
  signal in `threshold_analysis.py`; no keyword lists involved.
- **Search**: sentence embeddings win the hit-rate comparison decisively
  (0.94 vs 0.56) because they match meaning, not just words; zero-overlap
  queries like `money transfer` → the savings note then work.

See `RESULTS.md` for full numbers, error analysis, trade-offs and limitations.
