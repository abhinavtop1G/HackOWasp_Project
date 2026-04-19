"""
Greenwashing scorer — two modes:

  RULE-BASED (cloud/default): no torch needed, instant startup.
  TRANSFORMER (local dev):    fine-tuned DistilBERT, loaded lazily only
                               when the trained model directory exists AND
                               torch is importable.

All heavy imports are lazy (inside methods) so the module itself
imports in milliseconds regardless of mode.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from src import config

log = logging.getLogger("greenwashing_scorer")

# ── Rule-based markers ────────────────────────────────────────────────────────
_VAGUE = [
    r"\bworld.?class\b", r"\bbest.?in.?class\b", r"\bcommitted to\b",
    r"\bgenuine\b", r"\bmeaningful\b", r"\bsubstantial\b",
    r"\bhigh.?integrity\b", r"\bhigh.?impact\b", r"\blasting\b",
    r"\bdeep(ly)?\b", r"\bsustainab\w+\b", r"\bpositive impact\b",
    r"\btransformative\b", r"\bholistic\b", r"\brobust\b",
    r"\bpioneer\w*\b", r"\binnovat\w+\b", r"\beco.?friendly\b",
    r"\bclimate.?leader\w*\b", r"\bunprecedented\b", r"\bempow\w+\b",
    r"\btruly\b", r"\bgenuine\b",
]
_CRED = [
    r"\d+(?:\.\d+)?\s*%",
    r"\d[\d,]*\s*(?:ha|hectares?)",
    r"\d[\d,]*\s*(?:tonnes?|tons?|tCO2)",
    r"\bVM\d{4}\b", r"\bVCS\b", r"\bVCU\b",
    r"\b20\d{2}\b",
    r"\bbaseline\b", r"\bleakage\b", r"\badditionality\b",
    r"\bmonitoring\b", r"\bverif\w+\b", r"\baudit\w*\b",
    r"\bLiDAR\b", r"\bSentinel\b", r"\bLandsat\b",
    r"\bsample plot\b", r"\bbiomass\b",
]
_VP = [re.compile(p, re.IGNORECASE) for p in _VAGUE]
_CP = [re.compile(p, re.IGNORECASE) for p in _CRED]


def _rule_prob(text: str) -> float:
    v = sum(1 for p in _VP if p.search(text))
    c = sum(1 for p in _CP if p.search(text))
    net = v - c * 1.5
    prob = 1.0 / (1.0 + np.exp(-net * 0.8))
    return float(np.clip(prob, 0.05, 0.95))


class GreenwashingScorer:
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else config.GREENWASHING_MODEL_DIR
        self._mode = "rule_based"
        self._model = None
        self._tokenizer = None
        self._device = "cpu"

        # Only attempt transformer load if model dir exists
        if self.model_dir.exists():
            self._try_load_transformer()

        log.info(f"Greenwashing scorer mode: {self._mode}")

    def _try_load_transformer(self):
        """Lazy-load transformer — fails silently if torch not installed."""
        try:
            import torch  # noqa: F401  lazy import
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
            import torch as _torch
            self._device = "cuda" if _torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir
            ).to(self._device)
            self._model.eval()
            self._mode = "transformer"
        except ImportError:
            log.info("torch not installed — using rule-based scorer (cloud mode)")
        except Exception as e:
            log.warning(f"Could not load transformer ({e}) — using rule-based scorer")

    def score_claims(self, claims: list[str]) -> dict:
        if not claims:
            return {"aggregate_score": 0.0, "per_claim": [], "n_claims": 0}
        if self._mode == "transformer":
            return self._score_transformer(claims)
        return self._score_rule_based(claims)

    def _score_rule_based(self, claims: list[str]) -> dict:
        probs = [_rule_prob(t) for t in claims]
        per_claim = [
            {"text": t, "greenwashing_prob": p,
             "label": "greenwashing" if p >= 0.5 else "credible"}
            for t, p in zip(claims, probs)
        ]
        return {
            "aggregate_score": float(np.mean(probs) * 100),
            "per_claim": per_claim,
            "n_claims": len(claims),
        }

    def _score_transformer(self, claims: list[str]) -> dict:
        import torch
        enc = self._tokenizer(
            claims, truncation=True, padding=True,
            max_length=config.GREENWASHING_MAX_LEN, return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            probs = torch.softmax(
                self._model(**enc).logits, dim=-1
            )[:, 1].cpu().numpy()
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
