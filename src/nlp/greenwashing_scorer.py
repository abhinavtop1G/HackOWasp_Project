"""
Inference wrapper around the fine-tuned greenwashing classifier.

Usage:
    scorer = GreenwashingScorer()
    result = scorer.score_claims(["We care about sustainability.", "Forest cover: 87.3%."])
    # -> {'aggregate_score': 0.62, 'per_claim': [...]}

The aggregate score is the mean greenwashing probability across all claims,
scaled to 0-100. Higher = more greenwashy.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config

log = logging.getLogger("greenwashing_scorer")


class GreenwashingScorer:
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else config.GREENWASHING_MODEL_DIR
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Greenwashing model not found at {self.model_dir}. "
                f"Run: python -m src.nlp.train_greenwashing"
            )

        log.info(f"Loading greenwashing model from {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_claims(self, claims: list[str]) -> dict:
        """
        Score a list of claim sentences.
        Returns dict with:
          - aggregate_score: float in [0, 100], higher = more greenwashy overall
          - per_claim: list of {"text", "greenwashing_prob", "label"}
        """
        if not claims:
            return {"aggregate_score": 0.0, "per_claim": [], "n_claims": 0}

        enc = self.tokenizer(
            claims,
            truncation=True,
            padding=True,
            max_length=config.GREENWASHING_MAX_LEN,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # prob of class 1 (greenwashing)

        per_claim = []
        for text, p in zip(claims, probs):
            per_claim.append({
                "text": text,
                "greenwashing_prob": float(p),
                "label": "greenwashing" if p >= 0.5 else "credible",
            })

        aggregate = float(np.mean(probs) * 100)  # 0-100
        return {
            "aggregate_score": aggregate,
            "per_claim": per_claim,
            "n_claims": len(claims),
        }
