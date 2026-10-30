"""Baseline pipeline on a tiny in-test corpus (no download)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from sentiment.baseline import build_pipeline, save_baseline, train_baseline  # noqa: E402
from sentiment.predict import BaselinePredictor  # noqa: E402

POS = ["a wonderful moving film", "brilliant acting and a wonderful story",
       "loved it, brilliant", "wonderful, funny and warm", "a brilliant delight"]
NEG = ["a boring awful mess", "awful script and boring acting", "hated it, awful",
       "boring, dull and long", "an awful waste of time"]
TEXTS, LABELS = POS + NEG, [1] * len(POS) + [0] * len(NEG)


def test_defaults_match_v1() -> None:
    params = build_pipeline().get_params()
    assert params["tfidf__min_df"] == 5
    assert params["tfidf__ngram_range"] == (1, 2)
    assert params["tfidf__stop_words"] == "english"


def test_stop_words_drop_negation() -> None:
    """Documents the v1 trade-off: the English stop list removes 'not'."""
    with_stop = build_pipeline(min_df=1).named_steps["tfidf"].build_analyzer()
    without = build_pipeline(min_df=1, stop_words=None).named_steps["tfidf"].build_analyzer()
    assert "not" not in with_stop("not good")
    assert "not good" in without("not good")


def test_baseline_learns_tiny_corpus() -> None:
    pipe = train_baseline(TEXTS, LABELS, min_df=1)
    assert list(pipe.predict(TEXTS)) == LABELS
    prob = pipe.predict_proba(["wonderful and brilliant", "awful and boring"])[:, 1]
    assert prob[0] > 0.5 > prob[1]


def test_saved_baseline_round_trips_through_predictor(tmp_path: Path) -> None:
    path = save_baseline(train_baseline(TEXTS, LABELS, min_df=1), tmp_path / "b.joblib")
    preds = BaselinePredictor(path, version="t").predict(["brilliant", "awful"])
    assert [p.label for p in preds] == ["positive", "negative"]
    assert all(0.5 <= p.score <= 1.0 for p in preds)


def test_missing_artefact_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make baseline"):
        BaselinePredictor(tmp_path / "missing.joblib")
