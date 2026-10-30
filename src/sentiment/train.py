"""Fine-tune DistilBERT on IMDB with the Hugging Face Trainer.

Hyperparameters follow v1 (max length 128, batch 16, learning rate 3e-5, AdamW, linear
warm-up schedule, gradient accumulation, full fine-tuning). The Trainer replaces the v1
hand-written loop. It adds per-epoch validation, keeps the best checkpoint by F1, and
uses mixed precision on GPU.

The test split is never loaded here. Model selection uses a 10% validation slice of the
training split; :mod:`sentiment.evaluate` is the only place the test split is touched.

Run: ``python -m sentiment.train`` (GPU strongly recommended; see README for timings to record).
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sentiment import LABELS


@dataclass
class TrainConfig:
    """Fine-tuning settings. Defaults mirror v1 where v1 specified a value."""

    model_name: str = "distilbert-base-uncased"
    max_len: int = 128
    batch_size: int = 16
    grad_accum_steps: int = 2
    learning_rate: float = 3e-5
    epochs: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    val_frac: float = 0.1
    train_sample: int | None = None
    seed: int = 42
    output_dir: str = "models/distilbert-imdb"


def hf_compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Metric callback for the Trainer: logits to accuracy and F1 on the validation set."""
    from sentiment.evaluate import classification_metrics

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    m = classification_metrics(labels, preds)
    return {"accuracy": m["accuracy"], "f1": m["f1"], "macro_f1": m["macro_f1"]}


def train(cfg: TrainConfig) -> Path:
    """Fine-tune and save the best model to ``<output_dir>/final``."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from sentiment.data import load_imdb, stratified_sample, train_val_split

    set_seed(cfg.seed)
    train_df, _ = load_imdb()
    train_df = stratified_sample(train_df, cfg.train_sample, seed=cfg.seed)
    tr, val = train_val_split(train_df, val_frac=cfg.val_frac, seed=cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_len)

    train_ds = Dataset.from_pandas(tr).map(tokenize, batched=True, remove_columns=["text"])
    val_ds = Dataset.from_pandas(val).map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
        id2label=dict(enumerate(LABELS)),
        label2id={lab: i for i, lab in enumerate(LABELS)},
    )
    # Warm-up as a step count: works on transformers 4.x and 5.x (5.x dropped `warmup_ratio`).
    steps_per_epoch = math.ceil(len(train_ds) / (cfg.batch_size * cfg.grad_accum_steps))
    warmup_steps = max(1, round(cfg.warmup_ratio * steps_per_epoch * cfg.epochs))
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 4,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        num_train_epochs=cfg.epochs,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=cfg.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=hf_compute_metrics,
    )
    trainer.train()
    final = Path(cfg.output_dir) / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    val_metrics = trainer.evaluate()
    (final / "training_summary.json").write_text(json.dumps({
        "config": asdict(cfg),
        "n_train": len(tr),
        "n_val": len(val),
        "validation_metrics": val_metrics,
        "note": "Validation only. Test-set results come from sentiment.evaluate.",
    }, indent=2, default=str))
    return final


def main(argv: Sequence[str] | None = None) -> None:
    """CLI wrapper around :func:`train`."""
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT on IMDB.")
    defaults = TrainConfig()
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--grad-accum-steps", type=int, default=defaults.grad_accum_steps)
    p.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    p.add_argument("--max-len", type=int, default=defaults.max_len)
    p.add_argument("--train-sample", type=int, default=None, help="Quick run on n reviews.")
    p.add_argument("--output-dir", default=defaults.output_dir)
    p.add_argument("--seed", type=int, default=defaults.seed)
    a = p.parse_args(argv)
    cfg = TrainConfig(
        epochs=a.epochs, batch_size=a.batch_size, grad_accum_steps=a.grad_accum_steps,
        learning_rate=a.learning_rate, max_len=a.max_len, train_sample=a.train_sample,
        output_dir=a.output_dir, seed=a.seed,
    )
    print(f"Saved model to {train(cfg)}")


if __name__ == "__main__":
    main()
