# Model card: DistilBERT IMDB sentiment (sentiment-api v2)

*Last updated: 2026-10-29*

Format loosely follows Mitchell et al. (2019), "Model Cards for Model Reporting".

## Model details

| | |
|---|---|
| Model | `distilbert-base-uncased`, fully fine-tuned for binary sequence classification |
| Labels | `negative` (0), `positive` (1) |
| Input | English text, truncated to 128 WordPiece tokens |
| Output | label plus the probability of that label, and the positive-class probability |
| Training | Hugging Face Trainer, AdamW, learning rate 3e-5, batch 16 with 2-step gradient accumulation, linear schedule with 10% warm-up, 2 epochs, best epoch kept by validation F1 |
| Comparison model | TF-IDF (unigrams and bigrams, min_df 5, English stop words) plus logistic regression |
| Author | Sherie Lin |
| Licence | Code: MIT. Base model: Apache 2.0 (verify on the Hugging Face model page). Data: see below. |
| Version | Set by `SENTIMENT_MODEL_VERSION` and returned in every API response |

## Intended use

* Classifying the overall sentiment of English movie or entertainment reviews that are similar in style to IMDB user reviews.
* A worked example of a trained, evaluated, documented and containerised NLP model, for learning and portfolio purposes.
* Aggregate analysis (the share of positive reviews in a large set) rather than decisions about individual people.

## Out of scope

* Decisions about individuals: moderation, hiring, credit, or anything with legal or safety impact.
* Languages other than English, and domains far from film reviews (product support tickets, clinical notes, social media slang) without re-evaluation on that domain.
* Neutral, mixed or aspect-level sentiment. The model always picks one of two labels.
* Treating `score` as a calibrated probability. Calibration has not been measured (see next steps).

## Training data

IMDB Large Movie Review Dataset (Maas et al., 2011), loaded with Hugging Face `datasets` as `stanfordnlp/imdb`.

* 25,000 labelled training reviews and 25,000 labelled test reviews, each split 50/50 positive and negative. A further 50,000 unlabelled reviews are not used.
* Labels come from star ratings: 7 or more out of 10 is positive and 4 or less is negative. Neutral reviews (5 and 6) are excluded from the dataset, so the model has never seen them.
* At most 30 reviews per film, and the train and test sets use disjoint films.
* Pre-processing: HTML `<br />` tags removed, whitespace collapsed.
* Training uses 90% of the train split. The remaining 10% (stratified) is the validation set for choosing the epoch. The test split is used only for the final evaluation.
* Licence: the dataset page asks users to cite the paper; no formal licence is stated. **Verify** before any commercial use.

## Evaluation

Held-out IMDB test split (25,000 reviews), produced by `make evaluate`. 95% intervals are percentile bootstrap (1,000 resamples).

| Model | Accuracy (95% CI) | F1 (95% CI) | Macro F1 | ROC AUC | Throughput (texts/s, hardware) |
|---|---|---|---|---|---|
| DistilBERT (this model) | 0.883 (0.879-0.887) | 0.884 (0.880-0.889) | 0.883 | 0.954 | about 117, MacBook laptop |
| TF-IDF + logistic regression | 0.880 (0.876-0.885) | 0.881 (0.877-0.885) | 0.880 | 0.952 | about 15,600, same laptop |

Validation metrics (2,500 reviews, best of 2 epochs): DistilBERT accuracy 0.871, F1 0.874, loss 0.354. The baseline scored accuracy 0.884, F1 0.886 on the same slice.

**Reading:** the two models are within each other's 95% intervals on the test set, so this model is not measurably more accurate than the baseline, and it is about 130 times slower. The 128-token limit is the most likely reason (see limitations).

## Limitations and known failure modes

* **Truncation.** Only the first 128 tokens are read. Many IMDB reviews are longer, and reviewers often save their verdict for the end ("...but in the end I loved it"). Check `reports/errors_distilbert.csv` for this pattern.
* **Negation, sarcasm and irony** remain hard, especially in short texts.
* **No neutral class.** Mixed or lukewarm reviews are forced into positive or negative, often with high confidence.
* **Domain shift.** Accuracy on other kinds of text is unknown and likely lower.
* **Dataset bias.** IMDB reviewers are not a representative sample of people. The base model also inherits biases from its pre-training corpus (English Wikipedia and BookCorpus), which can show up as different scores for reviews that mention particular groups.
* **One run.** Results come from a single seed. Variation across seeds is not measured yet.
* **No accuracy gain over the baseline.** Within the 95% interval, the much cheaper TF-IDF model is as accurate on this test set.

## Ethical considerations

The model has low direct risk in its intended use. The main risk is reuse outside that scope, for example scoring individual people's posts or messages. The API returns a model version with every prediction so that outputs can be traced back to the model that produced them.

## Next steps

* Measure calibration (reliability curve, expected calibration error), and apply temperature scaling if needed.
* Compare against head-plus-tail truncation (first 64 and last 64 tokens) for long reviews.
* Repeat training over 3 seeds and report the spread.
* Evaluate on an out-of-domain set (for example product reviews) before claiming any general use.
