"""IMDB sentiment classification: data, baseline, fine-tuning, evaluation and prediction.

Heavy dependencies (``datasets``, ``transformers``, ``torch``) are imported inside the
functions that need them, so the API and tests can run without the training stack.
"""

__version__ = "2.0.0"
LABELS: tuple[str, str] = ("negative", "positive")
