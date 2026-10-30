from __future__ import annotations

import pandas as pd
import pytest

from sentiment.data import clean_text, stratified_sample, train_val_split


def _frame(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "text": [f"review {i}" for i in range(n)],
        "label": [i % 2 for i in range(n)],
    })


def test_clean_text_removes_html_breaks() -> None:
    assert clean_text("Great.<br /><br />Loved it.<BR>  Yes") == "Great. Loved it. Yes"


def test_train_val_split_is_disjoint_and_stratified() -> None:
    tr, val = train_val_split(_frame(), val_frac=0.2)
    assert len(tr) == 80 and len(val) == 20
    assert not set(tr["text"]) & set(val["text"])
    assert val["label"].mean() == 0.5


def test_train_val_split_validates_fraction() -> None:
    with pytest.raises(ValueError):
        train_val_split(_frame(), val_frac=1.5)


def test_stratified_sample_balances_classes() -> None:
    s = stratified_sample(_frame(), 40)
    assert len(s) == 40 and s["label"].mean() == 0.5


def test_stratified_sample_none_returns_all() -> None:
    assert len(stratified_sample(_frame(), None)) == 100
