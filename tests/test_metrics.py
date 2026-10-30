from __future__ import annotations

import numpy as np
import pytest

from sentiment.evaluate import (
    accuracy,
    bootstrap_ci,
    classification_metrics,
    confusion,
    f1_score,
    most_confident_errors,
    roc_auc,
)


def test_confusion_layout() -> None:
    assert confusion(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])) == [[1, 1], [1, 1]]


def test_metrics_known_values() -> None:
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([1, 1, 0, 0, 0, 1])
    m = classification_metrics(y, p)
    assert m["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["macro_f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["confusion_matrix"] == [[2, 1], [1, 2]]


def test_all_one_class_predictions_do_not_crash() -> None:
    m = classification_metrics(np.array([0, 1]), np.array([0, 0]))
    assert m["precision"] == 0.0 and m["f1"] == 0.0


def test_roc_auc_extremes_and_ties() -> None:
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert roc_auc(np.array([1, 1]), np.array([0.1, 0.2])) is None


def test_roc_auc_random_is_near_half() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    assert roc_auc(y, rng.random(5000)) == pytest.approx(0.5, abs=0.03)


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = np.where(rng.random(2000) < 0.85, y, 1 - y)
    lo, hi = bootstrap_ci(y, p, accuracy, n_boot=300)
    assert lo < accuracy(y, p) < hi
    assert hi - lo < 0.05
    lo_f, hi_f = bootstrap_ci(y, p, f1_score, n_boot=300)
    assert lo_f < f1_score(y, p) < hi_f


def test_most_confident_errors_sorted() -> None:
    errs = most_confident_errors(
        ["a", "b", "c", "d"], np.array([0, 0, 1, 1]), np.array([0.99, 0.6, 0.1, 0.9]), k=5
    )
    assert errs["text"].tolist() == ["a", "c", "b"]
