"""Prediction layer shared by the API and the evaluation script.

Every predictor exposes the same two methods, so the API does not care which model is
behind it and tests can swap in a fake.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from sentiment import LABELS

if TYPE_CHECKING:
    from sentiment.config import Settings


@dataclass(frozen=True)
class Prediction:
    """One prediction. ``score`` is the probability of the predicted label."""

    label: str
    score: float
    positive_probability: float


def to_predictions(pos_prob: np.ndarray, threshold: float = 0.5) -> list[Prediction]:
    """Turn positive-class probabilities into :class:`Prediction` objects."""
    out = []
    for p in np.asarray(pos_prob, dtype=float):
        positive = p >= threshold
        out.append(Prediction(
            label=LABELS[1] if positive else LABELS[0],
            score=round(float(p if positive else 1 - p), 6),
            positive_probability=round(float(p), 6),
        ))
    return out


class Predictor(Protocol):
    """Interface every model wrapper implements."""

    name: str
    version: str

    def positive_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Probability that each text is positive."""
        ...

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Labelled predictions for each text."""
        ...


class BaselinePredictor:
    """Wraps the saved TF-IDF + logistic regression pipeline."""

    name = "baseline"

    def __init__(self, path: str | Path, version: str = "dev") -> None:
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `make baseline` first.")
        self.pipeline = joblib.load(path)
        self.version = version

    def positive_proba(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.pipeline.predict_proba(list(texts))[:, 1])

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        return to_predictions(self.positive_proba(texts))


class TransformerPredictor:
    """Wraps a fine-tuned Hugging Face sequence-classification model on CPU or GPU.

    Inputs longer than ``max_len`` tokens are truncated, matching training.
    """

    name = "distilbert"

    def __init__(self, path: str | Path, max_len: int = 128, version: str = "dev") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `make train` first.")
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.model.eval()
        self.max_len = max_len
        self.version = version
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.pos_index = label2id.get(LABELS[1], 1)

    def positive_proba(self, texts: Sequence[str]) -> np.ndarray:
        enc = self.tokenizer(
            list(texts), truncation=True, max_length=self.max_len, padding=True,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.inference_mode():
            logits = self.model(**enc).logits
        probs = self._torch.softmax(logits, dim=-1)[:, self.pos_index]
        return probs.cpu().numpy()

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        return to_predictions(self.positive_proba(texts))


def load_predictor(settings: Settings) -> Predictor:
    """Build the predictor named in ``settings.predictor``."""
    if settings.predictor == "baseline":
        return BaselinePredictor(settings.baseline_path, version=settings.model_version)
    return TransformerPredictor(
        settings.model_path, max_len=settings.max_len, version=settings.model_version
    )
