"""One-line CLI to load the starter notes with their original IDs.

    python -m app.load_starter

Idempotent: notes whose IDs already exist are skipped. User-added notes use
IDs prefixed "U", so the reserved N001-N012 range is never reused. Starter
notes saved without a category get their prediction from the trained model,
so the same uncertainty flag logic applies.
"""

from __future__ import annotations

from app.classifier import load_classifier
from app.notes_store import NotesStore


def main() -> None:
    classifier = load_classifier()
    store = NotesStore()
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
    print(f"starter notes: added {summary['added']}, skipped {summary['skipped']}, "
          f"classified {filled} new notes")


if __name__ == "__main__":
    main()
