#!/usr/bin/env python3
"""Train the Smart Notes Assistant classifier.

Reproducible with one command:

    python train.py

What it does:
  1. Reads data/train.csv (156 rows) and data/validation.csv (50 rows).
  2. Fits two baselines: most-frequent-label and a keyword-rule predictor.
  3. Trains the final model: TF-IDF (word 1-2 grams, sublinear TF) +
     LogisticRegression, fitted on train.csv only.
  4. Prints accuracy and macro-F1 on validation.csv for the baselines and
     the trained model, plus a confusion matrix and the misclassified
     validation examples (used for the error analysis in RESULTS.md).
  5. Sweeps the uncertainty threshold on validation + out-of-scope notes
     (never used for training), picks one by a documented rule, and saves
     the model bundle (pipeline, labels, threshold) to
     models/classifier.joblib, which the app and /classify load once.

Validation labels are used only for measuring, never for fitting.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from app.textnorm import custom_stopwords

ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "classifier.joblib"

LABELS = ["Study", "Money", "Travel", "Shopping"]
RANDOM_STATE = 42

THRESHOLDS = np.round(np.arange(0.30, 0.90, 0.05), 2)

# Keyword rules for the second baseline (from CATEGORY_GUIDE.md definitions).
KEYWORD_RULES = {
    "Study": [
        "exam", "test", "study", "assignment", "homework", "quiz", "lecture",
        "course", "class", "proj", "chapter", "revise", "seminar", "thesis",
        "tutor", "syllabus",
    ],
    "Money": [
        "pay", "bill", "rent", "bank", "savings", "salary", "budget",
        "invoice", "money", "insurance", "installment", "instalment", "loan",
        "card", "emi", "subscribe", "deposit", "withdraw",
    ],
    "Travel": [
        "flight", "train", "hotel", "trip", "ticket", "itinerary", "travel",
        "airport", "station", "holiday", "vacation", "pack", "luggage",
        "departure", "check-in", "visa", "cruise", "bus",
    ],
    "Shopping": [
        "buy", "order", "purchase", "return", "deliver", "shop", "cart",
        "deal", "price", "discount", "groceries", "grocery", "clothes",
        "shoes", "gift", "checkout", "warranty",
    ],
}


def load_csv(path: Path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = (row.get("text") or "").strip()
            if not t:
                continue
            texts.append(t)
            labels.append((row.get("category") or "").strip())
    return texts, labels


def load_out_of_scope(path: Path):
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.reader(f)]
    texts = [r[0].strip() for r in rows[1:] if r and r[0].strip()]
    return texts


TOKEN_RE = re.compile(r"[a-z₹][a-z0-9₹]*")


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


class MostFrequentBaseline:
    """Always predicts the label that was most frequent in training data."""

    def fit(self, labels):
        counts = Counter(labels)
        top = max(counts.items(), key=lambda kv: (kv[1], LABELS.index(kv[0])))
        self.label_ = top[0]
        return self

    def predict(self, texts):
        return [self.label_] * len(texts)


class KeywordBaseline:
    """Scores each label by counting rule keywords contained in the text."""

    def predict(self, texts):
        preds = []
        for text in texts:
            low = text.lower()
            scores = {
                label: sum(1 for kw in kws if kw in low)
                for label, kws in KEYWORD_RULES.items()
            }
            best = max(LABELS, key=lambda lb: (scores[lb], -LABELS.index(lb)))
            preds.append(best)
        return preds


def build_pipeline() -> Pipeline:
    # Preprocessing chosen via experiments.py (5-fold CV on train.csv only):
    # removing generic stopwords beat word-only baseline (0.966 vs 0.947 CV
    # macro-F1), but sklearn's built-in 'english' list also deletes
    # task-relevant words such as "bill" (Money), "due" and "call". A custom
    # list minus those domain terms scored best (0.973), see app/textnorm.py.
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    stop_words=custom_stopwords,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )


def print_metrics(name: str, y_true, y_pred, show_errors=False, texts=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=LABELS)
    print(f"{name:<28} accuracy={acc:.3f}  macro-F1={f1:.3f}")
    if show_errors:
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        print(f"{name} confusion matrix (rows=true, cols=pred):")
        header = "              " + "  ".join(f"{lb:>8}" for lb in LABELS)
        print(header)
        for lb, row in zip(LABELS, cm):
            print(f"  {lb:<11} " + "  ".join(f"{v:>8}" for v in row))
        print(f"{name} misclassified examples:")
        n = 0
        for text, true, pred in zip(texts, y_true, y_pred):
            if true != pred:
                n += 1
                print(f"  [{n}] true={true} pred={pred} :: {text}")
    return acc, f1


def sweep_thresholds(val_proba_top, val_proba_margin, oos_proba_top, oos_proba_margin):
    """Print the flag-rate trade-off per uncertainty rule and pick one.

    A note is flagged uncertain when the rule fires; shares below are the
    fraction flagged (confidence/margin strictly below the cutoff).

    Signals:
      - maxproba : P(top class) < t
      - margin   : P(top) - P(2nd class) < m

    Selection rule (documented in RESULTS.md): among margin cutoffs with a
    validation flag rate <= 0.30, maximise the gap between the out-of-scope
    and validation flag rates; ties break towards the lower cutoff.
    Neither validation nor out-of-scope notes are used for training.
    """
    print("maxproba rule: uncertain when P(top) < t")
    print("threshold | oos_flagged | val_flagged")
    for t in THRESHOLDS:
        oos_rate = float(np.mean(oos_proba_top < t))
        val_rate = float(np.mean(val_proba_top < t))
        print(f"    {t:.2f}    |   {oos_rate:5.2f}     |   {val_rate:5.2f}")
    print()

    margin_cutoffs = np.round(np.arange(0.04, 0.20, 0.02), 2)
    print("margin rule: uncertain when P(top) - P(2nd) < m")
    print("cutoff    | oos_flagged | val_flagged")
    best = None
    for m in margin_cutoffs:
        oos_rate = float(np.mean(oos_proba_margin < m))
        val_rate = float(np.mean(val_proba_margin < m))
        gap = oos_rate - val_rate
        print(f"    {m:.2f}    |   {oos_rate:5.2f}     |   {val_rate:5.2f}")
        if val_rate <= 0.30:
            key = (gap, -float(m))
            if best is None or key > best[0]:
                best = (key, float(m), oos_rate, val_rate)
    if best is None:  # pragma: no cover - fallback if the cap rejects everything
        _, m, oos_rate, val_rate = ((0.0, -0.5), float(np.min(margin_cutoffs)), 0.0, 0.0)
    else:
        _, m, oos_rate, val_rate = best
    print(
        f"Chosen margin cutoff: {m:.2f} "
        f"(OOS flagged {oos_rate:.2f}, validation flagged {val_rate:.2f})"
    )
    return m


def main():
    train_texts, train_labels = load_csv(DATA / "train.csv")
    val_texts, val_labels = load_csv(DATA / "validation.csv")
    oos_texts = load_out_of_scope(DATA / "out_of_scope_examples.csv")
    print(f"Loaded train={len(train_texts)} val={len(val_texts)} oos={len(oos_texts)}\n")

    # --- Baselines ---
    mf = MostFrequentBaseline().fit(train_labels)
    print_metrics("Baseline: most-frequent", val_labels, mf.predict(val_texts))
    kw = KeywordBaseline()
    print_metrics(
        "Baseline: keyword rules", val_labels, kw.predict(val_texts),
        show_errors=True, texts=val_texts,
    )

    # --- Trained model ---
    pipe = build_pipeline()
    pipe.fit(train_texts, train_labels)
    val_pred = pipe.predict(val_texts)
    acc, f1 = print_metrics(
        "Model: TF-IDF + LogReg", val_labels, val_pred,
        show_errors=True, texts=val_texts,
    )

    # --- Uncertainty threshold sweep ---
    val_sorted = np.sort(pipe.predict_proba(val_texts), axis=1)
    oos_sorted = np.sort(pipe.predict_proba(oos_texts), axis=1)
    val_top, val_margin = val_sorted[:, -1], val_sorted[:, -1] - val_sorted[:, -2]
    oos_top, oos_margin = oos_sorted[:, -1], oos_sorted[:, -1] - oos_sorted[:, -2]
    margin_cutoff = sweep_thresholds(val_top, val_margin, oos_top, oos_margin)

    # --- Save the bundle the app loads ---
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "labels": LABELS,
            "uncertain_margin": margin_cutoff,
            "uncertain_rule": "margin: P(top) - P(2nd class) < cutoff",
            "model_name": (
                "TF-IDF (word 1-2 grams, sublinear, custom stopword list) "
                "+ LogisticRegression"
            ),
            "random_state": RANDOM_STATE,
        },
        MODEL_PATH,
    )
    print(f"Saved model bundle to {MODEL_PATH}")
    print(f"Validation accuracy={acc:.4f} macro-F1={f1:.4f} (numbers above, formatted)")


if __name__ == "__main__":
    main()
