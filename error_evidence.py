#!/usr/bin/env python3
"""Evidence for the validation mistakes analysed in RESULTS.md.

    python error_evidence.py[ --text "Some note"]

For each analysed validation mistake it prints:
  - the model's predicted label and per-class probabilities (incl. margin),
  - the TF-IDF feature contributions (coef * tfidf value) that pushed the
    prediction towards the wrong class and the true class,
  - whether the decisive tokens exist in the training vocabulary at all,
  - the nearest training examples by cosine similarity.

This proves the explanations in RESULTS.md are backed by the model itself.
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

ANALYSED = [
    "I must reach the railway station by 10 AM.",
    "Our return journey from Jaipur is on Wednesday.",
    "Put snacks and cleaning items in the cart.",
    "Check the credit card balance on Sunday.",
    "Need to exchange the size of a purchased item.",
]


def load_train():
    with open(ROOT / "data" / "train.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("text") or "").strip()]
    return [r["text"] for r in rows], [r["category"] for r in rows]


def evidence(bundle, texts, train_texts, train_labels):
    pipe = bundle["pipeline"]
    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    feat = tfidf.get_feature_names_out()
    vocab = set(feat)    # LogisticRegression.classes_ is alphabetically sorted and may differ from
    # the task order stored in bundle["labels"]; always index by classes_.
    coef_labels = list(clf.classes_)
    n_classes = len(coef_labels)

    x_all = tfidf.transform(train_texts + [t for t in texts])
    train_mat = x_all[: len(train_texts)]
    query_mat = x_all[len(train_texts):]

    for row_idx, (text, vec) in enumerate(zip(texts, query_mat)):
        print("=" * 88)
        print(f"NOTE: {text!r}")
        proba = pipe.predict_proba([text])[0]
        order = np.argsort(-proba)
        top, second = order[0], order[1]
        margin = proba[top] - proba[second]
        uncertain = margin < bundle["uncertain_margin"]
        print(
            f"predicted={coef_labels[top]}  uncertain={uncertain}  "
            + "  ".join(f"{coef_labels[i]}={proba[i]:.3f}" for i in order)
            + f"  margin={margin:.3f}"
        )

        vec_arr = np.asarray(vec.todense()).ravel()
        nz = np.nonzero(vec_arr)[0]
        contributions = {c: [] for c in range(n_classes)}
        for j in nz:
            for c in range(n_classes):
                contributions[c].append((vec_arr[j] * clf.coef_[c][j], feat[j]))
        parts = []
        for label, coef_idx in ((coef_labels[second], second), (coef_labels[top], top)):
            top_pos = sorted(contributions[coef_idx], reverse=True)[:4]
            parts.append(f"{label}: " + ", ".join(f"{w:+.2f}*{name!r}" for w, name in top_pos))
        print("top resource contributions -> " + " | ".join(parts))

        analyzer = tfidf.build_analyzer()
        tokens = analyzer(text)
        unknown_tokens = [t for t in tokens if t not in vocab]
        print(f"tokens/bigrams unseen in training: {sorted(set(unknown_tokens)) or '(none)'}")

        sims = np.asarray(
            (query_mat[[row_idx]] @ train_mat.T).todense()
        ).ravel()
        near = np.argsort(-sims)[:3]
        for i in near:
            print(f"  nearest train ({sims[i]:.2f}) [{train_labels[i]}] {train_texts[i]!r}")


def main():
    texts = list(ANALYSED)
    if len(sys.argv) > 1:
        texts = [" ".join(sys.argv[1:])]
    train_texts, train_labels = load_train()
    bundle = joblib.load(ROOT / "models" / "classifier.joblib")
    evidence(bundle, texts, train_texts, train_labels)


if __name__ == "__main__":
    main()
