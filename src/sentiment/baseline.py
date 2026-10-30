"""TF-IDF plus logistic regression baseline.

Defaults reproduce v1 (``min_df=5``, unigrams and bigrams, English stop words) so the
numbers stay comparable. Note that scikit-learn's English stop-word list contains
"not", "no" and "nor", so "not good" is reduced to "good". ``--keep-stop-words`` runs
the obvious ablation.

Run: ``python -m sentiment.baseline`` (writes ``models/baseline.joblib``).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline(
    min_df: int = 5,
    ngram_range: tuple[int, int] = (1, 2),
    stop_words: str | None = "english",
    max_features: int | None = None,
    sublinear_tf: bool = False,
    C: float = 1.0,
    seed: int = 42,
) -> Pipeline:
    """Create the unfitted baseline pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            min_df=min_df, ngram_range=ngram_range, stop_words=stop_words,
            max_features=max_features, sublinear_tf=sublinear_tf,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=C, random_state=seed)),
    ])


def train_baseline(
    texts: Sequence[str], labels: Sequence[int], **pipeline_kwargs: Any
) -> Pipeline:
    """Fit the baseline on raw texts and integer labels."""
    pipe = build_pipeline(**pipeline_kwargs)
    pipe.fit(list(texts), np.asarray(labels))
    return pipe


def save_baseline(pipe: Pipeline, path: str | Path) -> Path:
    """Persist the fitted pipeline with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    """Train on the IMDB train split (minus validation), report validation metrics, save."""
    from sentiment.data import load_imdb, stratified_sample, train_val_split
    from sentiment.evaluate import classification_metrics

    p = argparse.ArgumentParser(description="Train the TF-IDF + logistic regression baseline.")
    p.add_argument("--sample", type=int, default=None, help="Train on n rows (v1 used 10000).")
    p.add_argument("--keep-stop-words", action="store_true")
    p.add_argument("--out", default="models/baseline.joblib")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    train_df, _ = load_imdb()  # test split deliberately unused here
    train_df = stratified_sample(train_df, args.sample, seed=args.seed)
    tr, val = train_val_split(train_df, seed=args.seed)
    pipe = train_baseline(
        tr["text"], tr["label"], stop_words=None if args.keep_stop_words else "english",
        seed=args.seed,
    )
    prob = pipe.predict_proba(val["text"].tolist())[:, 1]
    metrics = classification_metrics(val["label"].to_numpy(), (prob >= 0.5).astype(int), prob)
    out = save_baseline(pipe, args.out)
    print(json.dumps({"split": "validation", "n_train": len(tr), **metrics}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
