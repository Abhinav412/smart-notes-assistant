#!/usr/bin/env python3
"""Pick preprocessing/model settings by cross-validation on TRAINING data only.

    python experiments.py

Never touches validation.csv. Compares TF-IDF + LogisticRegression variants
with 5-fold stratified CV on data/train.csv, motivated by the diagnosis that
stopword-ish n-grams ("to", "for", "our", "time for") dominate the Travel
class boundary and cause systematic misclassification.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from app.textnorm import custom_stopwords

ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LABELS = ["Study", "Money", "Travel", "Shopping"]
RANDOM_STATE = 42


def load_train():
    with open(ROOT / "data" / "train.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("text") or "").strip()]
    return [r["text"] for r in rows], [r["category"] for r in rows]


# sklearn's built-in 'english' list (Glasgow IR list) removes domain-relevant
# words for this task, e.g. "bill" (Money), "due", "call" (Travel), "put".
# Config E uses the sklearn list minus such task-specific terms (the same
# custom list lives in app/textnorm.py and is what train.py ships).
STOP_WORDS_SAFE = custom_stopwords

CONFIGS = {
    "A word(1,2) sublinear": dict(ngram_range=(1, 2), sublinear_tf=True),
    "B A + stop_words eng": dict(ngram_range=(1, 2), sublinear_tf=True, stop_words="english"),
    "C B + min_df=2": dict(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=2),
    "D C + C_logreg=2": dict(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=2),
    "E eng minus domain": dict(ngram_range=(1, 2), sublinear_tf=True, stop_words=STOP_WORDS_SAFE),
}


def cv_macro_f1(texts, labels, vectorizer_kwargs, c_logreg):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    macro = []
    accs = []
    for tr_idx, te_idx in skf.split(texts, labels):
        tr = [texts[i] for i in tr_idx]
        tr_y = [labels[i] for i in tr_idx]
        te = [texts[i] for i in te_idx]
        te_y = [labels[i] for i in te_idx]
        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectorizer_kwargs)),
                ("clf", LogisticRegression(max_iter=2000, C=c_logreg, random_state=RANDOM_STATE)),
            ]
        )
        pipe.fit(tr, tr_y)
        pred = pipe.predict(te)
        macro.append(f1_score(te_y, pred, average="macro", labels=LABELS))
        accs.append((np.array(pred) == np.array(te_y)).mean())
    return float(np.mean(macro)), float(np.mean(accs))


def main():
    texts, labels = load_train()
    print(f"Train rows: {len(texts)}\n")
    results = []
    for name, kwargs in CONFIGS.items():
        c = 2.0 if name.startswith("D") else 1.0
        macro, acc = cv_macro_f1(texts, labels, kwargs, c)
        results.append((name, acc, macro))
        print(f"{name:<24} CV accuracy={acc:.3f}  CV macro-F1={macro:.3f}")
    best = max(results, key=lambda r: r[2])
    print(f"\nBest by CV macro-F1: {best[0]} ({best[2]:.3f})")


if __name__ == "__main__":
    main()
