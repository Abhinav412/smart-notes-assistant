"""JSON-file persistence for notes.

Keeps the 12 supplied starter notes under their original IDs (N001-N012);
user-added notes always get IDs prefixed with "U" so reserved IDs are never
reused. Writes are atomic (temp file + replace) so a crash cannot corrupt
the store.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = Path(os.environ.get("NOTES_PATH", ROOT / "notes.json"))
STARTER_PATH = ROOT / "data" / "starter_notes.json"

RESERVED_PREFIX = "N"
USER_PREFIX = "U"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class NotesStore:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)
        self._notes: list[dict] = []
        self.load()

    # -- persistence ---------------------------------------------------
    def load(self) -> None:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            notes = data.get("notes", []) if isinstance(data, dict) else data
            self._notes = [n for n in notes if isinstance(n, dict) and n.get("id")]
        else:
            self._notes = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump({"notes": self._notes}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    # -- queries -------------------------------------------------------
    def all_notes(self) -> list[dict]:
        return list(self._notes)

    def get(self, note_id: str) -> Optional[dict]:
        return next((n for n in self._notes if n["id"] == note_id), None)

    def filtered(
        self,
        category: Optional[str] = None,
        uncertain: Optional[bool] = None,
    ) -> list[dict]:
        notes = self.all_notes()
        if category is not None:
            notes = [n for n in notes if n.get("category") == category]
        if uncertain is not None:
            notes = [n for n in notes if bool(n.get("uncertain")) is uncertain]
        return notes

    # -- mutations -----------------------------------------------------
    def _next_user_id(self) -> str:
        numbers = [
            int(n["id"][1:])
            for n in self._notes
            if n["id"].startswith(USER_PREFIX) and n["id"][1:].isdigit()
        ]
        return f"{USER_PREFIX}{max(numbers, default=0) + 1:04d}"

    def add(self, text: str, category: str, uncertain: bool) -> dict:
        note = {
            "id": self._next_user_id(),
            "text": text,
            "category": category,
            "uncertain": bool(uncertain),
            "created_at": _now(),
        }
        self._notes.append(note)
        self.save()
        return note

    def load_starter(self, path: Path = STARTER_PATH) -> dict:
        """Load data/starter_notes.json keeping its original IDs.

        Idempotent: an ID already present is skipped, so loading twice never
        duplicates notes. User IDs (prefixed "U") never collide with the
        reserved N001-N012 range.
        """
        with open(path, encoding="utf-8") as f:
            starter = json.load(f)
        added = skipped = 0
        for item in starter:
            note_id = item["id"]
            if self.get(note_id) is None:
                self._notes.append(
                    {
                        "id": note_id,
                        "text": item["text"],
                        "category": "",
                        "uncertain": False,
                        "created_at": _now(),
                    }
                )
                added += 1
            else:
                skipped += 1
        self.save()
        return {"added": added, "skipped": skipped}

    def needs_classification(self) -> Iterable[dict]:
        """Starter notes are stored with an empty category until classified."""
        for note in self.all_notes():
            if note.get("category") == "":
                yield note
