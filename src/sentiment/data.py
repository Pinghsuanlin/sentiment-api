"""Load and split the IMDB Large Movie Review dataset.

The official split is 25,000 train and 25,000 test reviews, balanced 50/50. The test split
is only ever used by :mod:`sentiment.evaluate`. Model selection uses a validation slice
carved out of the training split.
"""

from __future__ import annotations

import re

import pandas as pd

DATASET_NAME = "stanfordnlp/imdb"
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Replace the HTML line breaks that IMDB reviews contain and collapse whitespace."""
    return _WS_RE.sub(" ", _BR_RE.sub(" ", text)).strip()


def load_imdb(name: str = DATASET_NAME) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (or read from the Hugging Face cache) the labelled train and test splits.

    Returns:
        ``(train, test)`` DataFrames with columns ``text`` (cleaned) and ``label``
        (0 = negative, 1 = positive). The unlabelled ``unsupervised`` split is dropped.
    """
    from datasets import load_dataset

    ds = load_dataset(name)
    frames = []
    for split in ("train", "test"):
        df = ds[split].to_pandas()[["text", "label"]]
        df["text"] = df["text"].map(clean_text)
        frames.append(df.reset_index(drop=True))
    return frames[0], frames[1]


def stratified_sample(
    df: pd.DataFrame, n: int | None, seed: int = 42, label_col: str = "label"
) -> pd.DataFrame:
    """Class-balanced sample of about ``n`` rows (all rows if ``n`` is None or too big)."""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    frac = n / len(df)
    out = df.groupby(label_col, group_keys=False).sample(frac=frac, random_state=seed)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_val_split(
    df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42, label_col: str = "label"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation split. The two parts never share a row."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1")
    val = df.groupby(label_col, group_keys=False).sample(frac=val_frac, random_state=seed)
    train = df.drop(index=val.index)
    return train.reset_index(drop=True), val.reset_index(drop=True)
