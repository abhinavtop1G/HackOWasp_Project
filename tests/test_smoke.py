"""
Smoke test. Run:  python -m tests.test_smoke

Verifies:
  - Forest classifier loads and classifies a dummy tile
  - Greenwashing scorer loads and scores sample claims
  - Full pipeline runs on the synthetic hero project
"""
from __future__ import annotations

import logging
import sys

import numpy as np

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("smoke")


def test_forest_classifier():
    from src.satellite.forest_classifier import load_classifier, classify_tile
    clf = load_classifier()
    dummy = np.random.uniform(0, 3000, size=(4, 64, 64)).astype(np.float32)
    out = classify_tile(dummy, clf=clf)
    assert "forest_cover_pct_ml" in out
    assert 0 <= out["forest_cover_pct_ml"] <= 100
    log.info(f"  \u2713 forest classifier: {out['forest_cover_pct_ml']:.1f}% on random tile")


def test_greenwashing_scorer():
    from src.nlp.greenwashing_scorer import GreenwashingScorer
    scorer = GreenwashingScorer()
    claims = [
        "We are committed to a greener future.",
        "The project retained 87.3% of forest cover in 2022 per Sentinel-2 analysis.",
    ]
    out = scorer.score_claims(claims)
    assert out["n_claims"] == 2
    assert len(out["per_claim"]) == 2
    # The first should be more greenwashy than the second
    gw1 = out["per_claim"][0]["greenwashing_prob"]
    gw2 = out["per_claim"][1]["greenwashing_prob"]
    log.info(f"  vague: {gw1:.2f} | specific: {gw2:.2f}")
    assert gw1 > gw2, "Greenwashing classifier failed sanity check"
    log.info("  \u2713 greenwashing scorer sanity check passed")


def test_full_pipeline():
    import json
    from src.fusion.integrity_score import build_report
    from src.nlp.greenwashing_scorer import GreenwashingScorer
    from src.satellite.change_detection import compare_tiles
    from src.satellite.forest_classifier import load_classifier
    from src.satellite.planetary_computer_client import fetch_sentinel2_tile

    with open(config.HERO_PROJECTS_PATH) as f:
        projects = json.load(f)["projects"]
    project = next(p for p in projects if p["id"] == "SYN-DEMO-01")

    bands_b, _ = fetch_sentinel2_tile(project["lat"], project["lon"],
                                      project["date_start"], project["date_end"])
    bands_a, _ = fetch_sentinel2_tile(project["lat"], project["lon"],
                                      project["date_after_start"], project["date_after_end"])
    clf = load_classifier()
    sat = compare_tiles(bands_b, bands_a, clf=clf)

    scorer = GreenwashingScorer()
    gw = scorer.score_claims(project["summary_claims"])

    report = build_report(
        claimed_forest_cover_pct=project.get("claimed_forest_cover_pct"),
        measured_forest_cover_pct=sat["forest_after_pct"],
        forest_change_pct=sat["forest_change_pct"],
        greenwashing_scores=gw,
        n_numeric_claims=0,
        project_name=project["name"],
    )

    assert 0 <= report.integrity_score <= 100
    log.info(f"  \u2713 full pipeline: score={report.integrity_score:.1f} ({report.verdict})")


if __name__ == "__main__":
    try:
        log.info("[1/3] Testing forest classifier...")
        test_forest_classifier()
        log.info("[2/3] Testing greenwashing scorer...")
        test_greenwashing_scorer()
        log.info("[3/3] Testing full pipeline...")
        test_full_pipeline()
        log.info("ALL SMOKE TESTS PASSED.")
    except AssertionError as e:
        log.error(f"ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        log.exception(f"UNEXPECTED FAILURE: {e}")
        sys.exit(2)
