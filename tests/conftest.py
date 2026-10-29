import os
import sys
from pathlib import Path

# Make `app` importable when running pytest from the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Tests must never touch the developer's real notes.json.
os.environ.setdefault("NOTES_PATH", str(ROOT / "data" / "test_notes.json"))
