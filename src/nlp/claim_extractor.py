"""
Extract claim-like sentences from raw PDF text.

Two jobs:
  1. Identify sentences that are environmental/carbon "claims" (for greenwashing scoring)
  2. Extract structured numeric claims like "forest cover 87%", "145,320 tonnes CO2"
     (for cross-checking against satellite truth)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src import config


# Simple sentence splitter. Not perfect but works for well-formatted PDFs.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Numeric patterns we try to extract
FOREST_COVER_PATTERNS = [
    re.compile(r"(?:forest\s+cover|canopy\s+cover|tree\s+cover|retention).{0,40}?(\d{1,3}(?:\.\d+)?)\s*%", re.IGNORECASE),
    re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%\s*(?:forest\s+cover|canopy\s+cover|tree\s+cover|retention)", re.IGNORECASE),
    re.compile(r"retained\s+(\d{1,3}(?:\.\d+)?)\s*%", re.IGNORECASE),
]

HECTARES_PATTERN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:ha\b|hectares?)", re.IGNORECASE)
TONNES_PATTERN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:t|tonnes?|tons?)\s*(?:CO2|CO\u2082|carbon)", re.IGNORECASE)


@dataclass
class NumericClaim:
    kind: str          # "forest_cover_pct" | "area_hectares" | "co2_tonnes"
    value: float
    context: str       # surrounding sentence


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if len(s.strip()) > 10]


def is_environmental_claim(sentence: str) -> bool:
    s_lower = sentence.lower()
    return any(kw.lower() in s_lower for kw in config.CLAIM_KEYWORDS)


def extract_claim_sentences(text: str, max_claims: int = 40) -> list[str]:
    """Return sentences that look like environmental/carbon claims."""
    sentences = split_sentences(text)
    claims = [s for s in sentences if is_environmental_claim(s)]
    # Dedupe while preserving order
    seen, out = set(), []
    for c in claims:
        key = c.lower()[:80]
        if key not in seen:
            seen.add(key)
            out.append(c)
        if len(out) >= max_claims:
            break
    return out


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def extract_numeric_claims(text: str) -> list[NumericClaim]:
    out: list[NumericClaim] = []
    sentences = split_sentences(text)

    for sent in sentences:
        # Forest cover %
        for pat in FOREST_COVER_PATTERNS:
            m = pat.search(sent)
            if m:
                val = _to_float(m.group(1))
                if val is not None and 0 <= val <= 100:
                    out.append(NumericClaim("forest_cover_pct", val, sent[:200]))
                break

        # Hectares
        m = HECTARES_PATTERN.search(sent)
        if m:
            val = _to_float(m.group(1))
            if val is not None and val > 0:
                out.append(NumericClaim("area_hectares", val, sent[:200]))

        # Tonnes CO2
        m = TONNES_PATTERN.search(sent)
        if m:
            val = _to_float(m.group(1))
            if val is not None and val > 0:
                out.append(NumericClaim("co2_tonnes", val, sent[:200]))

    return out


def primary_forest_cover_claim(numeric_claims: list[NumericClaim]) -> Optional[float]:
    """Return the first forest-cover-% claim, which we treat as 'the' project claim."""
    for c in numeric_claims:
        if c.kind == "forest_cover_pct":
            return c.value
    return None
