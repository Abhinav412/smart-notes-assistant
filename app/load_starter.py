"""One-line CLI to load the starter notes with their original IDs.

    python -m app.load_starter

Idempotent: notes whose IDs already exist are skipped. User-added notes use
IDs prefixed "U", so the reserved N001-N012 range is never reused.

If the app server is running, this routes the request through it — the
server owns notes.json while it runs, and a direct file write from a second
process would be overwritten by the server's next save. Only when no server
answers does this tool write the store file directly (starter notes are then
classified on the next server start, see server.classify_pending()).
"""

from __future__ import annotations

import json
import time
from urllib.request import Request, urlopen

from app.classifier import load_classifier
from app.notes_store import NotesStore

SERVER_URL = "http://127.0.0.1:8000/api/notes/load-starter"


def try_running_server(timeout: float = 3.0, attempts: int = 3) -> dict | None:
    """Return the server's load summary, or None if no server is reachable.

    Retries briefly so a server busy with startup work (e.g. first embedding
    load) is still caught, rather than racing a direct file write against it.
    """
    for _ in range(attempts):
        try:
            req = Request(
                SERVER_URL,
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception:
            pass
        time.sleep(1.0)
    return None


def load_offline(path: str | None = None) -> dict:
    """Direct file access; only safe when no server is running."""
    classifier = load_classifier()
    store = NotesStore(path) if path else NotesStore()
    summary = store.load_starter()
    filled = 0
    for note in store.all_notes():
        if note.get("category") == "":
            result = classifier.classify(note["text"])
            note["category"] = result["category"]
            note["uncertain"] = result["uncertain"]
            filled += 1
    if filled:
        store.save()
    return {"added": summary["added"], "skipped": summary["skipped"], "classified": filled}


def main() -> None:
    via_server = try_running_server()
    if via_server is not None:
        print(
            f"starter notes (via running server): added {via_server['added']}, "
            f"skipped {via_server['skipped']}"
        )
        return
    result = load_offline()
    print(
        f"starter notes (direct file): added {result['added']}, "
        f"skipped {result['skipped']}, classified {result['classified']} new notes"
    )
    print(
        "NOTE: written directly to notes.json. Make sure the app server is NOT "
        "already running with an older store in memory, or use the UI button /"
        " the documented API call instead."
    )


if __name__ == "__main__":
    main()
