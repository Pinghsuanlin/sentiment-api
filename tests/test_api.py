"""API tests with a fake model: no weights, no torch, no network."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import create_app  # noqa: E402
from sentiment.config import get_settings  # noqa: E402
from sentiment.predict import Prediction, to_predictions  # noqa: E402


class FakePredictor:
    """Positive if the text contains 'good', else negative."""

    name = "fake"
    version = "test-1"

    def positive_proba(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([0.9 if "good" in t.lower() else 0.2 for t in texts])

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        return to_predictions(self.positive_proba(texts))


class BrokenPredictor(FakePredictor):
    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        raise RuntimeError("CUDA out of memory, internal detail")


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(loader=FakePredictor)) as c:
        yield c


def test_health_reports_loaded_model(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True and r.json()["predictor"] == "fake"


def test_predict_positive_and_negative(client: TestClient) -> None:
    pos = client.post("/predict", json={"text": "A really good film"}).json()
    neg = client.post("/predict", json={"text": "Dull and too long"}).json()
    assert pos["label"] == "positive" and pos["score"] == pytest.approx(0.9)
    assert neg["label"] == "negative" and neg["score"] == pytest.approx(0.8)
    assert neg["positive_probability"] == pytest.approx(0.2)
    assert pos["model_version"] == "test-1"


@pytest.mark.parametrize("payload", [
    {},
    {"text": ""},
    {"text": "   \n  "},
    {"text": 123},
    {"txt": "typo in field name"},
])
def test_predict_rejects_bad_input(client: TestClient, payload: dict) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_overlong_text(client: TestClient) -> None:
    too_long = "a" * (get_settings().max_text_chars + 1)
    assert client.post("/predict", json={"text": too_long}).status_code == 422


def test_batch_preserves_order(client: TestClient) -> None:
    r = client.post("/predict/batch", json={"texts": ["good", "bad", "so good"]})
    assert r.status_code == 200
    assert [p["label"] for p in r.json()["predictions"]] == ["positive", "negative", "positive"]


def test_batch_limits(client: TestClient) -> None:
    n = get_settings().max_batch_size
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422
    assert client.post("/predict/batch", json={"texts": ["x"] * (n + 1)}).status_code == 422
    assert client.post("/predict/batch", json={"texts": ["ok", ""]}).status_code == 422


def test_model_loaded_once() -> None:
    calls = []

    def loader() -> FakePredictor:
        calls.append(1)
        return FakePredictor()

    with TestClient(create_app(loader=loader)) as c:
        for _ in range(3):
            c.post("/predict", json={"text": "good"})
    assert len(calls) == 1


def test_missing_model_gives_503() -> None:
    def loader() -> FakePredictor:
        raise FileNotFoundError("no weights")

    with TestClient(create_app(loader=loader)) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json={"text": "good"}).status_code == 503


def test_internal_errors_are_not_leaked() -> None:
    with TestClient(create_app(loader=BrokenPredictor)) as c:
        r = c.post("/predict", json={"text": "good"})
    assert r.status_code == 500
    assert "CUDA" not in r.text
