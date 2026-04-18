"""
Fusion of satellite ground-truth, NLP greenwashing signal, and internal
consistency into a single Integrity Score (0-100).

Higher score = more trustworthy carbon project.

Three sub-scores, each 0-100:

  S_satellite (60%): How well the claimed forest cover matches measured.
      If claim doesn't exist, uses forest-cover change magnitude instead
      (large unexplained loss = low score).

  S_nlp (30%): 100 - aggregate greenwashing probability.
      Vague language shifts trust down.

  S_consistency (10%): Do the PDF's internal numbers self-consistent?
      Placeholder for now; can be extended.

Integrity = 0.6 * S_satellite + 0.3 * S_nlp + 0.1 * S_consistency
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from src import config


@dataclass
class IntegrityReport:
    integrity_score: float              # 0-100
    verdict: str                        # "HIGH INTEGRITY" | "MEDIUM" | "LOW INTEGRITY"
    satellite_subscore: float
    nlp_subscore: float
    consistency_subscore: float
    claimed_forest_cover_pct: Optional[float]
    measured_forest_cover_pct: Optional[float]
    discrepancy_pct: Optional[float]
    n_claims_analyzed: int
    n_greenwashing_flags: int
    top_greenwashing_claims: list[dict] = field(default_factory=list)
    explanation: str = ""
    methodology: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _verdict(score: float) -> str:
    if score >= 70:
        return "HIGH INTEGRITY"
    if score >= 40:
        return "MEDIUM INTEGRITY"
    return "LOW INTEGRITY"


def _satellite_subscore(
    claimed_pct: Optional[float],
    measured_pct: Optional[float],
    forest_change_pct: Optional[float],
) -> tuple[float, Optional[float]]:
    """
    Returns (subscore_0_100, discrepancy_pct).
    If a claimed % exists, score penalizes the |claim - measured| gap.
    Otherwise, score penalizes large absolute forest loss.
    """
    if claimed_pct is not None and measured_pct is not None:
        gap = abs(claimed_pct - measured_pct)
        # 0% gap -> 100 score; 50%+ gap -> 0 score; linear in between
        score = max(0.0, 100.0 - gap * 2.0)
        return score, gap

    if forest_change_pct is not None:
        # No claim to compare. Use magnitude of loss.
        # 0% loss -> 100; -30% -> 40; -50% -> 0
        if forest_change_pct >= 0:
            return 90.0, None  # gained forest, no complaints
        loss = -forest_change_pct
        score = max(0.0, 100.0 - loss * 2.0)
        return score, None

    return 50.0, None  # no data, no strong signal


def _nlp_subscore(aggregate_greenwashing_pct: float) -> float:
    """aggregate_greenwashing_pct is 0-100 where 100 means all claims are greenwashy."""
    return max(0.0, 100.0 - aggregate_greenwashing_pct)


def _consistency_subscore(n_claims: int, n_numeric: int) -> float:
    """
    Proxy: PDFs with many numeric claims (hectares, tonnes, dates) tend to be
    more credible than PDFs of pure marketing prose. Scale: more numerics = higher.
    """
    if n_claims == 0:
        return 50.0
    ratio = n_numeric / max(n_claims, 1)
    return float(min(100.0, 40.0 + 120.0 * ratio))  # 0 numerics -> 40, 0.5 ratio -> 100


def build_report(
    *,
    claimed_forest_cover_pct: Optional[float],
    measured_forest_cover_pct: Optional[float],
    forest_change_pct: Optional[float],
    greenwashing_scores: dict,        # output of GreenwashingScorer.score_claims
    n_numeric_claims: int,
    project_name: str = "",
) -> IntegrityReport:
    s_sat, gap = _satellite_subscore(
        claimed_forest_cover_pct, measured_forest_cover_pct, forest_change_pct
    )
    s_nlp = _nlp_subscore(greenwashing_scores.get("aggregate_score", 0.0))
    s_cons = _consistency_subscore(
        greenwashing_scores.get("n_claims", 0), n_numeric_claims
    )

    integrity = (
        config.FUSION_WEIGHT_SATELLITE * s_sat
        + config.FUSION_WEIGHT_NLP * s_nlp
        + config.FUSION_WEIGHT_CONSISTENCY * s_cons
    )

    per_claim = greenwashing_scores.get("per_claim", [])
    top_gw = sorted(
        [c for c in per_claim if c["greenwashing_prob"] >= 0.5],
        key=lambda c: -c["greenwashing_prob"],
    )[:5]

    # Explanation text
    parts = []
    proj = f"{project_name}: " if project_name else ""
    if claimed_forest_cover_pct is not None and measured_forest_cover_pct is not None:
        parts.append(
            f"{proj}Project claimed {claimed_forest_cover_pct:.1f}% forest retention; "
            f"Sentinel-2 analysis measured {measured_forest_cover_pct:.1f}% "
            f"(discrepancy: {gap:.1f} percentage points)."
        )
    elif forest_change_pct is not None:
        sign = "+" if forest_change_pct >= 0 else ""
        parts.append(
            f"{proj}Satellite analysis shows {sign}{forest_change_pct:.1f}% forest cover "
            f"change over the project period."
        )
    gw_pct = greenwashing_scores.get("aggregate_score", 0.0)
    n_flags = len(top_gw)
    parts.append(
        f"NLP screen flagged {n_flags} claims as greenwashing-style "
        f"(aggregate vagueness: {gw_pct:.1f}/100 across {greenwashing_scores.get('n_claims', 0)} claims)."
    )
    parts.append(f"Integrity Score: {integrity:.1f}/100 \u2014 {_verdict(integrity)}.")
    explanation = " ".join(parts)

    methodology = {
        "fusion_weights": {
            "satellite": config.FUSION_WEIGHT_SATELLITE,
            "nlp": config.FUSION_WEIGHT_NLP,
            "consistency": config.FUSION_WEIGHT_CONSISTENCY,
        },
        "satellite_source": "Sentinel-2 L2A via Microsoft Planetary Computer, NDVI + trained logistic classifier",
        "nlp_model": f"{config.GREENWASHING_BASE_MODEL} fine-tuned on labeled environmental claims",
        "version": config.PROVENANCE_VERSION,
    }

    return IntegrityReport(
        integrity_score=float(integrity),
        verdict=_verdict(integrity),
        satellite_subscore=float(s_sat),
        nlp_subscore=float(s_nlp),
        consistency_subscore=float(s_cons),
        claimed_forest_cover_pct=claimed_forest_cover_pct,
        measured_forest_cover_pct=measured_forest_cover_pct,
        discrepancy_pct=gap,
        n_claims_analyzed=greenwashing_scores.get("n_claims", 0),
        n_greenwashing_flags=n_flags,
        top_greenwashing_claims=top_gw,
        explanation=explanation,
        methodology=methodology,
    )
