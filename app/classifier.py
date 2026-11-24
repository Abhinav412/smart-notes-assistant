"""Loads the trained model bundle once and classifies note text.

The bundle (models/classifier.joblib) is produced by `python train.py` and
contains the fitted pipeline, the label list and the uncertainty cutoff.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning

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
    # A bundle saved by a different scikit-learn version may unpickle "fine"
    # (only issuing a warning) and then crash later inside predict_proba —
    # which surfaced as an unexplained HTTP 500 on the first classification
    # request. Turn the warning into an error here so a mismatch fails at
    # startup with a clear message instead.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        try:
            bundle = joblib.load(MODEL_PATH)
        except InconsistentVersionWarning as exc:
            raise RuntimeError(
                "Model bundle / environment mismatch: the bundle in "
                f"{MODEL_PATH} was saved by a different scikit-learn "
                f"version than the one installed here ({exc}). Serve the "
                "app from the same environment that ran 'python train.py'."
            ) from exc
    return NoteClassifier(bundle)
