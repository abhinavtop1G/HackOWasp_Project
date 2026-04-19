"""
Greenwashing scorer — two modes:
  1. TRAINED MODE: uses fine-tuned DistilBERT (when model exists locally)
  2. RULE-BASED MODE: fallback for cloud deployment (no training needed)

Rule-based mode scores each sentence by counting vague/credible linguistic
markers identified in the training data. It produces similar output format
to the ML model so the rest of the pipeline is unchanged.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from src import config

log = logging.getLogger("greenwashing_scorer")

# ── Rule-based markers ─────────────────────────────────────────────────────────
_VAGUE_MARKERS = [
    r"\bworld.?class\b", r"\bbest.?in.?class\b", r"\bcommitted to\b",
    r"\bgenuine\b", r"\bmeaningful\b", r"\bsubstantial\b", r"\bhigh.?integrity\b",
    r"\bhigh.?impact\b", r"\bpride\b", r"\blasting\b", r"\bdeep(ly)?\b",
    r"\bsustainab\w+\b", r"\bpositive impact\b", r"\btransformative\b",
    r"\bcomprehensive\b", r"\bholistic\b", r"\brobust\b", r"\bpioneer\w*\b",
    r"\bresponsib\w+\b", r"\binnovat\w+\b", r"\beco.?friendly\b",
    r"\bclimate.?leader\w*\b", r"\bunprecedented\b", r"\bempow\w+\b",
    r"\bexemplif\w+\b", r"\btruly\b", r"\bfully\b", r"\bgenuine\b",
]

_CREDIBLE_MARKERS = [
    r"\d+(?:\.\d+)?\s*%",           # any percentage
    r"\d[\d,]*\s*(?:ha|hectares?)",  # hectares
    r"\d[\d,]*\s*(?:tonnes?|tons?|tCO2)",  # tonnes
    r"\bVM\d{4}\b", r"\bVCS\b", r"\bVCU\b",  # methodology IDs
    r"\bGold Standard\b", r"\bVerra\b",
    r"\b20\d{2}\b",                  # year reference
    r"\bbaseline\b", r"\bleakage\b", r"\badditionality\b",
    r"\bmonitoring\b", r"\bverif\w+\b", r"\baudit\w*\b",
    r"\bLiDAR\b", r"\bSentinel\b", r"\bLandsat\b",
    r"\bsample plot\b", r"\bbiomass\b",
]

_VAGUE_PATS = [re.compile(p, re.IGNORECASE) for p in _VAGUE_MARKERS]
_CRED_PATS  = [re.compile(p, re.IGNORECASE) for p in _CREDIBLE_MARKERS]


def _rule_based_prob(text: str) -> float:
    """Score 0.0 (credible) to 1.0 (greenwashing) using lexical rules."""
    vague = sum(1 for p in _VAGUE_PATS if p.search(text))
    cred  = sum(1 for p in _CRED_PATS  if p.search(text))
    # Simple logistic-style combination
    net = vague - cred * 1.5   # credible markers weighted more
    prob = 1.0 / (1.0 + np.exp(-net * 0.8))
    return float(np.clip(prob, 0.05, 0.95))


# ── Main scorer class ──────────────────────────────────────────────────────────
class GreenwashingScorer:
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else config.GREENWASHING_MODEL_DIR
        self._mode = "rule_based"
        self._model = None
        self._tokenizer = None

        if self.model_dir.exists():
            try:
                self._load_transformer()
                self._mode = "transformer"
                log.info("Greenwashing scorer: transformer mode (fine-tuned DistilBERT)")
            except Exception as e:
                log.warning(f"Could not load transformer model ({e}). Using rule-based fallback.")
        else:
            log.info("Greenwashing scorer: rule-based mode (no trained model found — cloud deployment)")

    def _load_transformer(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir
        ).to(self._device)
        self._model.eval()

    def score_claims(self, claims: list[str]) -> dict:
        """
        Score claims. Returns same structure regardless of mode.
        """
        if not claims:
            return {"aggregate_score": 0.0, "per_claim": [], "n_claims": 0}

        if self._mode == "transformer":
            return self._score_transformer(claims)
        return self._score_rule_based(claims)

    def _score_rule_based(self, claims: list[str]) -> dict:
        per_claim = []
        probs = []
        for text in claims:
            p = _rule_based_prob(text)
            probs.append(p)
            per_claim.append({
                "text": text,
                "greenwashing_prob": p,
                "label": "greenwashing" if p >= 0.5 else "credible",
            })
        aggregate = float(np.mean(probs) * 100)
        return {"aggregate_score": aggregate, "per_claim": per_claim, "n_claims": len(claims)}

    def _score_transformer(self, claims: list[str]) -> dict:
        import torch
        enc = self._tokenizer(
            claims, truncation=True, padding=True,
            max_length=config.GREENWASHING_MAX_LEN, return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        per_claim = [
            {"text": t, "greenwashing_prob": float(p),
             "label": "greenwashing" if p >= 0.5 else "credible"}
            for t, p in zip(claims, probs)
        ]
        return {
            "aggregate_score": float(np.mean(probs) * 100),
            "per_claim": per_claim,
            "n_claims": len(claims),
        }
