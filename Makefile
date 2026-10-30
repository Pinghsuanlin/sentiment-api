PY ?= python
IMAGE ?= sentiment-api

.PHONY: install install-train lint test baseline train evaluate serve docker-build docker-run clean

install:        ## Core + dev (enough for tests and the baseline)
	$(PY) -m pip install -e ".[dev]"

install-train:  ## Add datasets, transformers, torch, matplotlib
	$(PY) -m pip install -e ".[dev,train,serve]"

lint:
	ruff check src api tests

test:           ## Offline tests: fake model, tiny in-test corpus
	$(PY) -m pytest

baseline:       ## TF-IDF + logistic regression -> models/baseline.joblib
	$(PY) -m sentiment.baseline

train:          ## Fine-tune DistilBERT -> models/distilbert-imdb/final (GPU recommended)
	$(PY) -m sentiment.train

evaluate:       ## Held-out test set -> reports/metrics.json, confusion_*.png, errors_*.csv
	$(PY) -m sentiment.evaluate --models baseline distilbert

serve:
	uvicorn api.main:app --reload --port 8000

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 -v "$(PWD)/models:/app/models:ro" $(IMAGE)

clean:
	rm -rf .pytest_cache .ruff_cache .coverage reports/*.json reports/*.png reports/*.csv
