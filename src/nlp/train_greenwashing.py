"""
Fine-tune DistilBERT as a binary classifier:
  label 1 = vague / unsubstantiated / greenwashing claim
  label 0 = specific / verifiable / credible claim

Run:
    python -m src.nlp.train_greenwashing

This saves the fine-tuned model + tokenizer to config.GREENWASHING_MODEL_DIR
and prints a classification report on the held-out test set.

Training time: ~3-6 min on a T4 GPU, ~15-20 min on CPU for the provided dataset.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# torch/transformers imported lazily — not available on cloud deployment
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
except ImportError:
    torch = None  # type: ignore

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("train_greenwashing")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class ClaimsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_samples(path: Path) -> tuple[list[str], list[int]]:
    with open(path) as f:
        data = json.load(f)
    samples = data["samples"]
    texts = [s["text"] for s in samples]
    labels = [int(s["label"]) for s in samples]
    log.info(f"Loaded {len(texts)} samples ({sum(labels)} greenwashing, {len(labels) - sum(labels)} credible)")
    return texts, labels


def evaluate(model, loader, device) -> tuple[float, str]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    report = classification_report(
        all_labels, all_preds,
        target_names=["credible", "greenwashing"],
        zero_division=0,
    )
    return f1, report


def train():
    texts, labels = load_samples(config.TRAINING_CLAIMS_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    log.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.GREENWASHING_BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.GREENWASHING_BASE_MODEL, num_labels=2
    ).to(device)

    train_ds = ClaimsDataset(X_train, y_train, tokenizer, config.GREENWASHING_MAX_LEN)
    test_ds = ClaimsDataset(X_test, y_test, tokenizer, config.GREENWASHING_MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=config.GREENWASHING_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.GREENWASHING_BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.GREENWASHING_LR, weight_decay=0.01)
    total_steps = len(train_loader) * config.GREENWASHING_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_f1 = -1.0
    for epoch in range(config.GREENWASHING_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_t = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_t)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += out.loss.item()

        f1, report = evaluate(model, test_loader, device)
        log.info(f"Epoch {epoch + 1}/{config.GREENWASHING_EPOCHS} | loss={epoch_loss / len(train_loader):.4f} | test F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            config.GREENWASHING_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(config.GREENWASHING_MODEL_DIR)
            tokenizer.save_pretrained(config.GREENWASHING_MODEL_DIR)
            log.info(f"  \u2713 saved new best model to {config.GREENWASHING_MODEL_DIR}")

    log.info("\n=== Final classification report ===\n" + report)
    log.info(f"Best test F1: {best_f1:.4f}")
    return best_f1


if __name__ == "__main__":
    train()
