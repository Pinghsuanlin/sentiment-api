# Sentiment API (v2)

A movie-review sentiment classifier taken all the way from data to a running service. It trains a TF-IDF baseline and a fine-tuned DistilBERT on IMDB, measures both on the held-out test set, documents the model in a model card, and serves it through a typed FastAPI service in a non-root Docker container.

## Why

v1 trained a model and wrapped it in Flask, but the headline transformer number was "expected F1 above 90%". It was never measured. A model you cannot measure is not ready to deploy. v2 is about doing the unglamorous parts properly: a clean train/validation/test split, numbers with confidence intervals, an honest model card, input validation, tests, and CI.

## What's new in v2

| Area | v1 | v2 |
|---|---|---|
| Data | Manual CSV | Hugging Face `datasets` (`stanfordnlp/imdb`), HTML breaks cleaned |
| Splits | 10k sample | Official 25k train / 25k test. Stratified 10% validation from train for epoch selection. Test touched only by `evaluate.py` |
| Baseline | TF-IDF + logistic regression on a 10k sample | Same settings, trained on the full train split, scored on the full test split. Optional ablation that keeps stop words |
| Transformer | Hand-written training loop | HF `Trainer`: same hyperparameters, per-epoch validation, best checkpoint by F1, mixed precision on GPU |
| Evaluation | Baseline only; DistilBERT never measured | `reports/metrics.json` for both models: accuracy, precision, recall, F1, macro F1, ROC AUC, 95% bootstrap intervals, confusion matrix plot, and the 50 most confident mistakes |
| Documentation | README | README plus [MODEL_CARD.md](MODEL_CARD.md) (intended use, out of scope, limitations) |
| API | Flask `/predict` | FastAPI: pydantic request and response models, `/predict`, `/predict/batch`, `/health`, model loaded once at startup, 503 when the model is missing, internal errors never leaked |
| Container | `python:3.9-slim`, root | `python:3.11-slim`, CPU-only torch, non-root user, health check, weights mounted at runtime |
| Tests and CI | None | Offline pytest (fake model, tiny in-test corpus), ruff, and a Docker build-and-health job in GitHub Actions |

## Architecture

```
            Hugging Face datasets: stanfordnlp/imdb
                    |                         |
               train (25k)                test (25k)  <- only evaluate.py reads this
                    |                         |
       stratified 90/10 split                 |
          |                |                  |
     train (22.5k)    validation (2.5k)       |
          |                |                  |
   +------+-------+        |                  |
   |              |        |                  |
baseline.py    train.py <--+ (epoch picked    |
TF-IDF + LR    DistilBERT     by val F1)      |
   |              |                           |
models/        models/distilbert-imdb/final   |
baseline.joblib   |                           |
   +------+-------+                           |
          |                                   |
     predict.py (one Predictor interface) ----+--> evaluate.py --> reports/metrics.json
          |                                                        reports/confusion_*.png
     api/main.py (FastAPI)                                         reports/errors_*.csv
          |
     Docker: python:3.11-slim, non-root, models/ mounted read-only
```

## Design decisions and trade-offs

**The test set is used once, at the end.** Choosing the best epoch on the test set would make the reported number optimistic. A stratified 10% validation slice of the training split does that job instead. `train.py` never loads the test split.

**Keep v1's baseline settings, then question them.** The baseline keeps v1's settings (min_df 5, unigrams and bigrams, English stop words) so the two versions can be compared. But scikit-learn's English stop-word list contains "not", "no" and "nor", so "not good" turns into "good", which is a poor trade for sentiment. A test documents this. `python -m sentiment.baseline --keep-stop-words` runs the ablation.

**128 tokens: cheaper, but blind to endings.** v1 used 128 tokens, and so does v2. That makes training several times cheaper than at 512 tokens. The cost is that the model only reads the opening of long reviews, and the verdict is often at the end. The model card lists this as the first limitation, and head-plus-tail truncation is the planned experiment.

**Trainer over a custom loop.** v1's hand-written loop worked, but it had no evaluation or checkpointing. The HF Trainer runs the same recipe (AdamW, linear warm-up, gradient accumulation) and adds per-epoch validation, keep-best-checkpoint, and fp16, with less code to maintain.

**Intervals, not just point estimates.** With 25,000 test reviews, the 95% interval on accuracy is roughly plus or minus 0.4 points. Two models within that band should not be called different. The bootstrap is in plain NumPy, so it is easy to check.

**One `Predictor` interface.** The baseline and DistilBERT both expose `positive_proba` and `predict`. The API, the evaluation script and the tests all depend on the interface, not the model. That is why the API tests run with a fake model and no torch.

**Load once, serve many.** The model loads in FastAPI's lifespan handler. The endpoints are plain `def` functions, so FastAPI runs them in its thread pool and a slow forward pass does not block the event loop. If the weights are missing, the service still starts, but `/health` returns 503 and `/predict` returns 503, so an orchestrator will not route traffic to it.

**Strict input limits.** Empty, whitespace-only, non-string and over-long texts are rejected with 422 before they reach the model. Batch size and total characters are capped. Unexpected model errors return a generic 500, and the details go to the log, not the client.

**Weights mounted, not baked.** The image does not contain the model, so it stays smaller and one image can serve any checkpoint. The Dockerfile shows the one-line change to bake weights in when you want a self-contained artefact.

## Data

IMDB Large Movie Review Dataset: 50,000 labelled reviews (25k train, 25k test, balanced), plus 50,000 unlabelled reviews that are not used.

* Source: https://ai.stanford.edu/~amaas/data/sentiment/ and https://huggingface.co/datasets/stanfordnlp/imdb
* Citation: Maas, Daly, Pham, Huang, Ng and Potts (2011), "Learning Word Vectors for Sentiment Analysis", ACL 2011.
* Licence: no formal licence is stated; the authors ask that you cite the paper. **Verify** before any commercial use.

The data downloads automatically into the Hugging Face cache on first run. Nothing is committed.

## Tools

Python 3.10+, scikit-learn, Hugging Face `datasets` and `transformers`, PyTorch, FastAPI, pydantic v2, pydantic-settings, uvicorn, matplotlib, pytest, ruff, Docker, GitHub Actions.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # or use uv
make install-train

make baseline        # CPU is fine                        -> models/baseline.joblib
make train           # GPU strongly recommended            -> models/distilbert-imdb/final
make evaluate        # full 25k test set, both models      -> reports/

make serve           # http://localhost:8000/docs
curl -s localhost:8000/predict -H 'content-type: application/json' \
     -d '{"text": "Slow start, but the last hour is wonderful."}'
curl -s localhost:8000/predict/batch -H 'content-type: application/json' \
     -d '{"texts": ["Loved it.", "Two hours I will not get back."]}'
```

To serve the baseline instead, set `SENTIMENT_PREDICTOR=baseline` (no torch needed for inference). For a quick end-to-end check before a full run, use `python -m sentiment.train --train-sample 2000 --epochs 1` and `python -m sentiment.evaluate --sample 2000`. Numbers from a sample are for smoke-testing only; do not report them.

**Docker**

```bash
make docker-build
make docker-run      # mounts ./models read-only into the container
```

**Optional: ONNX export and quantisation (not benchmarked here)**

For faster CPU serving you can export to ONNX with Hugging Face Optimum (`optimum-cli export onnx --model models/distilbert-imdb/final --task text-classification onnx/`), then apply dynamic int8 quantisation with ONNX Runtime. Alternatively, use `torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)` in PyTorch. Either option usually shrinks the model and speeds up CPU inference, at some cost in accuracy. Re-run `make evaluate` on the quantised model and compare against the full-precision numbers before switching. Neither path is wired into this repo yet.

## Tests

```bash
make test
```

The tests run offline, with no model weights and no torch:

* **API** with a fake predictor: health, single and batch prediction, label and score shape, order preserved in batches; 422 for empty, whitespace-only, non-string, misspelt-field, over-long and over-sized batch inputs; model loaded exactly once; 503 when the weights are missing; internal error text never returned to the client.
* **Baseline** on a ten-sentence corpus written in the test: learns it, round-trips through joblib and `BaselinePredictor`, defaults match v1, and the stop-word negation trade-off is pinned down.
* **Metrics** against hand-computed values: confusion layout, precision, recall, F1, macro F1, ROC AUC (perfect, inverted, ties, random), and bootstrap intervals.
* **Data**: HTML clean-up, a disjoint and stratified train/validation split, and balanced sampling.

## Results

Full 25,000-review test set, evaluated once after training (29 Oct 2026). 95% intervals are percentile bootstrap (1,000 resamples). Source: `reports/metrics.json`.

| Model | Train data | Test data | Accuracy (95% CI) | F1 (95% CI) | ROC AUC |
|---|---|---|---|---|---|
| v2 TF-IDF + LR | 22.5k (train minus val) | 25k held-out test | 0.880 (0.876-0.885) | 0.881 (0.877-0.885) | 0.952 |
| v2 DistilBERT, 128 tokens | 22.5k (train minus val) | 25k held-out test | 0.883 (0.879-0.887) | 0.884 (0.880-0.889) | 0.954 |
| v1 TF-IDF + LR (from the v1 README) | 10k sample | v1 split of the sample | 0.87 | 0.87 | not reported |
| v1 DistilBERT (from the v1 README) | not reported | not reported | never measured | "expected above 90%" (not a result) | not reported |

The v1 baseline was trained and scored on a 10k sample, so it is not directly comparable to the v2 numbers on the full test set.

**What the numbers say.**

* **DistilBERT does not beat the baseline.** Accuracy is 0.883 against 0.880, and the 95% intervals overlap, so the two models are statistically tied on this test set. On the validation slice the baseline was slightly ahead (F1 0.886 against 0.874).
* **The baseline is about 130 times faster.** On the same laptop (a MacBook) it scored about 15,600 reviews per second; DistilBERT scored about 117.
* **v1's "expected above 90%" did not hold.** Measured properly, the 128-token DistilBERT reaches 88.3%.
* **Likely cause, not yet tested: truncation.** At 128 tokens the model reads roughly the first 90-100 words, while the median review is longer, and reviewers often give their verdict at the end. Head-plus-tail truncation (first 64 and last 64 tokens) is the next experiment.
* **Decision on this evidence:** serve the baseline (`SENTIMENT_PREDICTOR=baseline`). It is as accurate, far cheaper, and needs no torch. Revisit if the truncation experiment opens a gap wider than the confidence interval.

## Challenges

* **Turning "expected" into "measured".** This mostly meant setting up the split discipline first, so that the final number is one nobody tuned against.
* **Testing an ML service without the ML.** Putting a single `Predictor` interface between the API and the model made it possible to test the API fully with a ten-line fake. CI does not need to download roughly 250 MB of weights.
* **Keeping the image lean.** The default PyTorch wheel bundles CUDA. Installing the CPU wheel first, and mounting the weights at runtime, keeps the serving image to what inference needs.

## Limitations and next steps

See [MODEL_CARD.md](MODEL_CARD.md) for the full list. In short: English film reviews only, binary labels with no neutral class, 128-token truncation, calibration not measured, and a single training seed.

Next: head-plus-tail truncation for long reviews, calibration with temperature scaling, three-seed runs, an out-of-domain test set, ONNX or int8 serving with a measured accuracy cost, and request logging with a drift check on input length and predicted-label share.

## Licence

MIT, see [LICENSE](LICENSE). The dataset and the base model have their own terms (see Data and the model card).
