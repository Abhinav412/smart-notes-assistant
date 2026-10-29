#!/usr/bin/env python3
"""Compare uncertainty signals on validation + out-of-scope data.

    python threshold_analysis.py

Part D of TASK.md allows using validation + data/out_of_scope_examples.csv to
choose the uncertainty approach and threshold (neither is trained on). This
script prints, per candidate signal and cutoff, the share of out-of-scope and
validation notes flagged, so the trade-off can be picked transparently.

Signals compared:
  - maxprob  : uncertain when P(top class) < t
  - margin   : uncertain when P(top) - P(2nd) < m
  - both     : uncertain when maxprob < t AND margin < m (AND-rule)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_val():
    with open(ROOT / "data" / "validation.csv", encoding="utf-8-sig") as f:
        return [r["text"].strip() for r in csv.DictReader(f) if r.get("text", "").strip()]


def load_oos():
    with open(ROOT / "data" / "out_of_scope_examples.csv", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    return [r[0].strip() for r in rows[1:] if r and r[0].strip()]


def main():
    bundle = joblib.load(ROOT / "models" / "classifier.joblib")
    pipe = bundle["pipeline"]
    val = load_val()
    oos = load_oos()
    pv = pipe.predict_proba(val)
    po = pipe.predict_proba(oos)

    def stats(p):
        srt = np.sort(p, axis=1)
        return srt[:, -1], srt[:, -1] - srt[:, -2]

    vmax, vmargin = stats(pv)
    omax, omargin = stats(po)

    def rate(cond):
        return float(cond.mean())

    print(f"{'signal':<26}{'cutoff':>8}{'OOS flagged':>13}{'val flagged':>13}")
    for t in [0.35, 0.40, 0.45, 0.50]:
        print(f"{'maxproba <':<26}{t:>8}{rate(omax < t):>13.2f}{rate(vmax < t):>13.2f}")
    for m in [0.05, 0.08, 0.10, 0.12, 0.15]:
        print(f"{'margin <':<26}{m:>8}{rate(omargin < m):>13.2f}{rate(vmargin < m):>13.2f}")
    for t, m in [(0.45, 0.10), (0.50, 0.12), (0.40, 0.08), (0.50, 0.15), (0.55, 0.15), (0.55, 0.20)]:
        flag_o = (omax < t) & (omargin < m)
        flag_v = (vmax < t) & (vmargin < m)
        print(f"{'maxproba & margin AND':<26}{f'{t:.2f}/{m:.2f}':>8}{rate(flag_o):>13.2f}{rate(flag_v):>13.2f}")


if __name__ == "__main__":
    main()
