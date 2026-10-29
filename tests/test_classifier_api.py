"""API tests: POST /classify contract + note flows (needs the trained model).

Run `python train.py` first; these tests are skipped otherwise.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.server import app

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "classifier.joblib"
LABELS = {"Study", "Money", "Travel", "Shopping"}

pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason="run `python train.py` to create the model first"
)


@pytest.fixture(scope="module")
def client():
    # deterministic, isolated notes file for the whole test module
    path = ROOT / "data" / "test_notes.json"
    if path.exists():
        path.unlink()
    with TestClient(app) as c:  # 'with' runs startup (loads model once)
        yield c


def test_classify_valid_note(client):
    res = client.post("/classify", json={"text": "Pay my electricity bill tomorrow."})
    assert res.status_code == 200
    body = res.json()
    assert body["category"] in LABELS
    assert isinstance(body["uncertain"], bool)


def test_classify_returns_best_guess_even_when_unsure(client):
    res = client.post("/classify", json={"text": "Call grandma this weekend."})
    assert res.status_code == 200
    body = res.json()
    assert body["category"] in LABELS
    assert body["uncertain"] is True  # out-of-scope notes are flagged


INVALID_BODIES = [
    {"text": "   "},
    {"text": ""},
    {"note": "Pay rent"},
    {"text": 42},
    {"text": None},
    {},
]


@pytest.mark.parametrize("body", INVALID_BODIES)
def test_classify_rejects_invalid_input(client, body):
    res = client.post("/classify", json=body)
    assert 400 <= res.status_code < 500


def test_classify_semantics_on_familiar_phrases(client):
    # Sanity: phrases of this shape dominate training data, so the trained
    # model should get them right; a regression here means a broken mapping.
    expected = {
        "Pay the electricity bill by the 28th.": "Money",
        "Confirm the bus seat for Pune.": "Travel",
        "Finish the mathematics assignment tonight.": "Study",
        "Return the defective headphones before Thursday.": "Shopping",
    }
    for text, label in expected.items():
        body = client.post("/classify", json={"text": text}).json()
        assert body["category"] == label, f"{text!r} -> {body['category']}"


def test_malformed_json_body_returns_4xx(client):
    res = client.post(
        "/classify",
        content=b'{"text": ',
        headers={"Content-Type": "application/json"},
    )
    assert 400 <= res.status_code < 500


def test_server_survives_invalid_requests(client):
    for body in INVALID_BODIES:
        client.post("/classify", json=body)
    res = client.post("/classify", json={"text": "Revise for my biology exam."})
    assert res.status_code == 200
    assert res.json()["category"] in LABELS


def test_same_input_same_output(client):
    first = client.post("/classify", json={"text": "Buy laptop stand."}).json()
    second = client.post("/classify", json={"text": "Buy laptop stand."}).json()
    assert first == second


def test_add_note_assigns_user_id_and_classification(client):
    res = client.post("/api/notes", json={"text": "Water the balcony plants."})
    assert res.status_code == 201
    body = res.json()
    assert body["id"].startswith("U")
    assert body["id"] not in {f"N{i:03d}" for i in range(1, 13)}
    assert body["category"] in LABELS
    assert isinstance(body["uncertain"], bool)


def test_starter_load_via_api_is_idempotent(client):
    first = client.post("/api/notes/load-starter").json()
    second = client.post("/api/notes/load-starter").json()
    assert first["added"] + len(client.get("/api/notes").json()["notes"]) > 0
    assert second["added"] == 0 and second["skipped"] == 12
    notes = client.get("/api/notes").json()["notes"]
    ids = [n["id"] for n in notes]
    assert "N001" in ids and "N012" in ids
    for n in notes:
        if n["id"].startswith("N"):
            assert n["category"] in LABELS


def test_search_finds_saved_note(client):
    client.post("/api/notes", json={"text": "Ship the parcel to Mysuru branch office."})
    res = client.get("/api/search", params={"q": "Mysuru branch"})
    assert res.status_code == 200
    results = res.json()["results"]
    assert results, "expected hits for the saved note"
    assert "Mysuru branch" in results[0]["text"]


def test_filter_by_category(client):
    notes = client.get("/api/notes", params={"category": "Study"}).json()["notes"]
    assert notes
    assert all(n["category"] == "Study" for n in notes)
