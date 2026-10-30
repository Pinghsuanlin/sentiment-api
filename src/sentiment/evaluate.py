"""Held-out test evaluation for the baseline and the fine-tuned transformer.

Metric functions are plain NumPy so they are easy to test and have no hidden defaults.
``main`` scores each model on the 25,000-review IMDB test split and writes:

* ``reports/metrics.json``             metrics, 95% bootstrap intervals, throughput
* ``reports/confusion_<model>.png``    confusion matrix plot (needs matplotlib)
* ``reports/errors_<model>.csv``       the most confident mistakes, for error analysis

Run: ``python -m sentiment.evaluate --models baseline distilbert``
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentiment import LABELS


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    """2x2 confusion matrix ``[[tn, fp], [fn, tp]]`` (rows = true label)."""
    t = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    return [
        [int(((t == 0) & (p == 0)).sum()), int(((t == 0) & (p == 1)).sum())],
        [int(((t == 1) & (p == 0)).sum()), int(((t == 1) & (p == 1)).sum())],
    ]


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """ROC AUC via the rank-sum (Mann-Whitney) identity, with average ranks for ties."""
    t = np.asarray(y_true).astype(int)
    n_pos, n_neg = int(t.sum()), int(len(t) - t.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(np.asarray(y_score, dtype=float)).rank().to_numpy()
    return float((ranks[t == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _prf(cm: list[list[int]]) -> tuple[float, float, float, float]:
    (tn, fp), (fn, tp) = cm
    n = tn + fp + fn + tp
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, prec, rec, f1


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> dict[str, Any]:
    """Accuracy, precision, recall and F1 (positive class), macro F1, confusion, ROC AUC."""
    cm = confusion(y_true, y_pred)
    acc, prec, rec, f1 = _prf(cm)
    # F1 for the negative class: swap the roles in the confusion matrix
    _, _, _, f1_neg = _prf([[cm[1][1], cm[1][0]], [cm[0][1], cm[0][0]]])
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "macro_f1": round((f1 + f1_neg) / 2, 4),
        "confusion_matrix": cm,
        "confusion_labels": list(LABELS),
    }
    if y_score is not None:
        auc = roc_auc(y_true, y_score)
        out["roc_auc"] = None if auc is None else round(auc, 4)
    return out


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for any metric of (y_true, y_pred)."""
    rng = np.random.default_rng(seed)
    t, p = np.asarray(y_true), np.asarray(y_pred)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(t), len(t))
        stats[i] = metric(t[idx], p[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return round(float(lo), 4), round(float(hi), 4)


def accuracy(t: np.ndarray, p: np.ndarray) -> float:
    """Share of correct predictions."""
    return float((np.asarray(t) == np.asarray(p)).mean())


def f1_score(t: np.ndarray, p: np.ndarray) -> float:
    """F1 for the positive class."""
    return _prf(confusion(t, p))[3]


def most_confident_errors(
    texts: Sequence[str], y_true: np.ndarray, pos_prob: np.ndarray, k: int = 50
) -> pd.DataFrame:
    """The ``k`` wrong predictions the model was most sure about."""
    df = pd.DataFrame({"text": list(texts), "label": y_true, "positive_prob": pos_prob})
    df["pred"] = (df["positive_prob"] >= 0.5).astype(int)
    df["confidence"] = np.where(df["pred"] == 1, df["positive_prob"], 1 - df["positive_prob"])
    wrong = df[df["pred"] != df["label"]]
    return wrong.sort_values("confidence", ascending=False).head(k)


def plot_confusion(cm: list[list[int]], title: str, path: str | Path) -> Path:
    """Save a labelled confusion-matrix heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(arr, cmap="Blues")
    for (i, j), v in np.ndenumerate(arr):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > arr.max() / 2 else "black")
    ax.set_xticks([0, 1], labels=list(LABELS))
    ax.set_yticks([0, 1], labels=list(LABELS))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def evaluate_model(
    name: str,
    predict_pos_prob: Callable[[list[str]], np.ndarray],
    texts: list[str],
    labels: np.ndarray,
    reports_dir: Path,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Score one model on the test texts and write its plot and error file."""
    t0 = time.perf_counter()
    probs = np.concatenate([
        predict_pos_prob(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)
    ])
    elapsed = time.perf_counter() - t0
    preds = (probs >= 0.5).astype(int)
    metrics = classification_metrics(labels, preds, probs)
    metrics["accuracy_ci95"] = bootstrap_ci(labels, preds, accuracy)
    metrics["f1_ci95"] = bootstrap_ci(labels, preds, f1_score)
    metrics["throughput_texts_per_s"] = round(len(texts) / elapsed, 1) if elapsed else None
    most_confident_errors(texts, labels, probs).to_csv(reports_dir / f"errors_{name}.csv",
                                                       index=False)
    try:
        plot_confusion(metrics["confusion_matrix"], f"{name} (IMDB test)",
                       reports_dir / f"confusion_{name}.png")
    except ImportError:
        metrics["plot"] = "matplotlib not installed; plot skipped"
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate the requested models on the IMDB test split."""
    from sentiment.config import Settings
    from sentiment.data import load_imdb, stratified_sample
    from sentiment.predict import load_predictor

    p = argparse.ArgumentParser(description="Evaluate models on the held-out IMDB test split.")
    p.add_argument("--models", nargs="+", default=["baseline", "distilbert"],
                   choices=["baseline", "distilbert"])
    p.add_argument("--sample", type=int, default=None,
                   help="Score a stratified subset of the test split (quick check only).")
    p.add_argument("--reports-dir", default="reports")
    args = p.parse_args(argv)

    settings = Settings()
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    _, test = load_imdb()
    test = stratified_sample(test, args.sample)
    texts, labels = test["text"].tolist(), test["label"].to_numpy()

    results: dict[str, Any] = {
        "dataset": "stanfordnlp/imdb",
        "split": "test",
        "n_test": len(texts),
        "full_test_set": args.sample is None,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": {},
    }
    for name in args.models:
        predictor = load_predictor(settings.model_copy(update={"predictor": name}))
        results["models"][name] = {
            "artifact": str(settings.baseline_path if name == "baseline" else settings.model_path),
            **evaluate_model(name, predictor.positive_proba, texts, labels, reports),
        }
        m = results["models"][name]
        print(f"{name}: accuracy={m['accuracy']} {m['accuracy_ci95']} f1={m['f1']} "
              f"auc={m.get('roc_auc')}")
    (reports / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {reports / 'metrics.json'}")


if __name__ == "__main__":
    main()
