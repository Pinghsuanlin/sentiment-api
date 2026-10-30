"""Sentiment API.

Endpoints:

* ``GET  /health``         liveness plus which model is loaded
* ``POST /predict``        one review
* ``POST /predict/batch``  up to ``SENTIMENT_MAX_BATCH_SIZE`` reviews in one call

The model is loaded once, in the lifespan handler, and shared across requests. The
endpoints are plain ``def`` functions, so FastAPI runs inference in its thread pool
and a slow forward pass does not block the event loop.

Run: ``uvicorn api.main:app --port 8000``
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from sentiment import __version__
from sentiment.config import get_settings
from sentiment.predict import Predictor, load_predictor

logger = logging.getLogger("sentiment.api")
settings = get_settings()

ReviewText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=settings.max_text_chars),
]


class PredictRequest(BaseModel):
    """A single review."""

    model_config = ConfigDict(json_schema_extra={"examples": [{"text": "A warm, funny film."}]})

    text: ReviewText


class BatchRequest(BaseModel):
    """Several reviews. Order is preserved in the response."""

    texts: list[ReviewText] = Field(min_length=1, max_length=settings.max_batch_size)

    @field_validator("texts")
    @classmethod
    def _total_size(cls, v: list[str]) -> list[str]:
        if sum(len(t) for t in v) > settings.max_text_chars * 4:
            raise ValueError("batch is too large in total characters")
        return v


class PredictionOut(BaseModel):
    """One prediction. ``score`` is the probability of the returned label."""

    label: Literal["negative", "positive"]
    score: float = Field(ge=0.0, le=1.0)
    positive_probability: float = Field(ge=0.0, le=1.0)


class PredictResponse(PredictionOut):
    """Single prediction plus model metadata."""

    model_config = ConfigDict(protected_namespaces=())

    model_version: str


class BatchResponse(BaseModel):
    """Batch predictions plus model metadata."""

    model_config = ConfigDict(protected_namespaces=())

    predictions: list[PredictionOut]
    model_version: str


class HealthResponse(BaseModel):
    """Liveness and model status."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    predictor: str | None
    model_version: str | None
    api_version: str


def get_predictor(request: Request) -> Predictor:
    """Dependency: the predictor loaded at startup, or 503 if loading failed."""
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor


PredictorDep = Annotated[Predictor, Depends(get_predictor)]


def create_app(loader: Callable[[], Predictor] | None = None) -> FastAPI:
    """Build the app. Tests pass a ``loader`` that returns a fake predictor."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = None
        try:
            app.state.predictor = (loader or (lambda: load_predictor(settings)))()
            logger.info("Loaded predictor %s", app.state.predictor.name)
        except FileNotFoundError:
            logger.exception("Model artefact missing; /health will report degraded")
        yield
        app.state.predictor = None

    app = FastAPI(title="Sentiment API", version=__version__, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request, response: Response) -> HealthResponse:
        p = request.app.state.predictor
        if p is None:
            response.status_code = 503  # lets container health checks see a missing model
        return HealthResponse(
            status="ok" if p is not None else "degraded",
            model_loaded=p is not None,
            predictor=getattr(p, "name", None),
            model_version=getattr(p, "version", None),
            api_version=__version__,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest, predictor: PredictorDep) -> PredictResponse:
        pred = _run(predictor, [req.text])[0]
        return PredictResponse(**pred.model_dump(), model_version=predictor.version)

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(req: BatchRequest, predictor: PredictorDep) -> BatchResponse:
        return BatchResponse(predictions=_run(predictor, req.texts),
                             model_version=predictor.version)

    return app


def _run(predictor: Predictor, texts: list[str]) -> list[PredictionOut]:
    """Call the model and convert results; hide internal errors from clients."""
    try:
        preds = predictor.predict(texts)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    return [
        PredictionOut(label=p.label, score=p.score, positive_probability=p.positive_probability)
        for p in preds
    ]


app = create_app()
