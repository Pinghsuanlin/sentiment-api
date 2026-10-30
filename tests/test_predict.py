from __future__ import annotations

import numpy as np
import pytest

from sentiment.predict import to_predictions


def test_to_predictions_threshold_and_score() -> None:
    preds = to_predictions(np.array([0.9, 0.5, 0.1]))
    assert [p.label for p in preds] == ["positive", "positive", "negative"]
    assert preds[0].score == pytest.approx(0.9)
    assert preds[2].score == pytest.approx(0.9)
    assert preds[2].positive_probability == pytest.approx(0.1)


def test_to_predictions_empty() -> None:
    assert to_predictions(np.array([])) == []
