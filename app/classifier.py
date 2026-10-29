"""Loads the trained model bundle once and classifies note text.

The bundle (models/classifier.joblib) is produced by `python train.py` and
contains the fitted pipeline, the label list and the uncertainty cutoff.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "classifier.joblib"


class NoteClassifier:
    def __init__(self, bundle: dict):
        self.pipeline = bundle["pipeline"]
        self.labels = bundle["labels"]
        # predict_proba columns follow the estimator's classes_ order, which
        # sklearn sorts alphabetically and which may differ from bundle
        # ["labels"]. Always map through the estimator's own class names.
        self.classes = [str(c) for c in self.pipeline.classes_]
        self.margin_cutoff = bundle["uncertain_margin"]
        self.description = bundle.get("model_name", "unknown model")

    @property
    def uncertain_rule(self) -> str:
        return f"{RULE_TEXT} (margin cutoff {self.margin_cutoff:.2f})"

    def classify(self, text: str) -> dict:
        proba = self.pipeline.predict_proba([text])[0]
        top, second = np.argsort(-proba)[:2]
        margin = float(proba[top] - proba[second])
        return {
            "category": self.classes[int(top)],
            "uncertain": bool(margin < self.margin_cutoff),
            "confidence": float(proba[top]),
            "margin": margin,
        }


RULE_TEXT = "P(top) - P(2nd class) < cutoff"


def load_classifier() -> NoteClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {MODEL_PATH}. Run 'python train.py' first."
        )
    return NoteClassifier(joblib.load(MODEL_PATH))
