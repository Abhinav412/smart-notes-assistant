"""Persistence tests for the JSON note store (no server needed)."""

import json

import pytest

from app.notes_store import NotesStore


def make_store(tmp_path):
    store = NotesStore(tmp_path / "notes.json")
    assert store.all_notes() == []
    return store


def test_save_and_reload_round_trip(tmp_path):
    store = make_store(tmp_path)
    note = store.add("Pay my electricity bill.", "Money", False)
    reloaded = NotesStore(tmp_path / "notes.json")
    again = reloaded.get(note["id"])
    assert again is not None
    assert again["text"] == "Pay my electricity bill."
    assert again["category"] == "Money"
    assert again["uncertain"] is False


def test_starter_load_is_idempotent(tmp_path):
    store = make_store(tmp_path)
    first = store.load_starter()
    assert first["added"] == 12
    second = store.load_starter()
    assert second["added"] == 0
    assert second["skipped"] == 12
    ids = [n["id"] for n in store.all_notes()]
    assert sorted(ids) == sorted({i for i in ids})  # no duplicates
    assert "N001" in ids and "N012" in ids


def test_user_ids_never_reuse_reserved_ids(tmp_path):
    store = make_store(tmp_path)
    store.load_starter()
    note = store.add("Call grandma this weekend.", "Travel", True)
    assert note["id"] not in {f"N{i:03d}" for i in range(1, 13)}
    assert note["id"].startswith("U")
    assert store.get(note["id"]) ["uncertain"] is True


def test_filtered_by_category_and_uncertain(tmp_path):
    store = make_store(tmp_path)
    store.add("Bring the samples to class.", "Study", False)
    store.add("Call grandma this weekend.", "Travel", True)
    assert [n["id"] for n in store.filtered(category="Study")] == ["U0001"]
    assert [n["id"] for n in store.filtered(uncertain=True)] == ["U0002"]


def test_store_file_is_valid_json(tmp_path):
    store = make_store(tmp_path)
    store.add("hello", "Money", False)
    data = json.loads((tmp_path / "notes.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data["notes"][0]["text"] == "hello"
