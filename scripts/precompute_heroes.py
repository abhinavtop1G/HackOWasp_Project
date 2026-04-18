"""
Pre-compute full VerifEarth analysis for all hero projects and cache.

Run BEFORE the demo so the satellite fetch (the slow part) happens once over
a reliable connection and the demo just replays cached results.

    python -m scripts.precompute_heroes
"""
from __future__ import annotations

import json
import logging
import time

from src import config
from src.fusion.integrity_score import build_report
from src.nlp.greenwashing_scorer import GreenwashingScorer
from src.satellite.change_detection import compare_tiles, render_comparison_png
from src.satellite.forest_classifier import load_classifier
from src.satellite.planetary_computer_client import fetch_sentinel2_tile

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("precompute")


def main():
    with open(config.HERO_PROJECTS_PATH) as f:
        data = json.load(f)

    scorer = GreenwashingScorer()
    clf = load_classifier()

    for project in data["projects"]:
        pid = project["id"]
        log.info(f"--- Processing {pid}: {project['name']} ---")
        t0 = time.time()

        bands_before, meta_b = fetch_sentinel2_tile(
            project["lat"], project["lon"],
            project["date_start"], project["date_end"],
            synthetic_forest_bias=project.get("synthetic_forest_bias_before"),
        )
        bands_after, meta_a = fetch_sentinel2_tile(
            project["lat"], project["lon"],
            project["date_after_start"], project["date_after_end"],
            synthetic_forest_bias=project.get("synthetic_forest_bias_after"),
        )
        sat = compare_tiles(bands_before, bands_after, clf=clf)

        gw = scorer.score_claims(project["summary_claims"])

        report = build_report(
            claimed_forest_cover_pct=project.get("claimed_forest_cover_pct"),
            measured_forest_cover_pct=sat["forest_after_pct"],
            forest_change_pct=sat["forest_change_pct"],
            greenwashing_scores=gw,
            n_numeric_claims=0,
            project_name=project["name"],
        )

        png_path = config.CACHED_RESULTS_DIR / f"{pid}.png"
        render_comparison_png(sat, png_path, title=project["name"])

        out = report.to_dict()
        out.update({
            "project_id": pid,
            "project_name": project["name"],
            "satellite_details": {
                "forest_before_pct": sat["forest_before_pct"],
                "forest_after_pct": sat["forest_after_pct"],
                "forest_change_pct": sat["forest_change_pct"],
                "ndvi_before_mean": sat["ndvi_before_mean"],
                "ndvi_after_mean": sat["ndvi_after_mean"],
                "source_before": meta_b,
                "source_after": meta_a,
            },
            "png_path": str(png_path),
            "timestamp": int(time.time()),
        })

        json_path = config.CACHED_RESULTS_DIR / f"{pid}.json"
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)

        log.info(
            f"  \u2713 {pid}: score={out['integrity_score']:.1f} verdict={out['verdict']} "
            f"(took {time.time() - t0:.1f}s)"
        )

    log.info("All hero projects precomputed.")


if __name__ == "__main__":
    main()
