"""Verify each hero project produces a sensible Integrity Score under offline fallback."""
import json
import sys
sys.path.insert(0, ".")

from src.satellite.forest_classifier import load_classifier
from src.satellite.planetary_computer_client import fetch_sentinel2_tile
from src.satellite.change_detection import compare_tiles
from src.fusion.integrity_score import build_report

with open("data/hero_projects.json") as f:
    projects = json.load(f)["projects"]

clf = load_classifier()

# Fake NLP scores: Kariba-style vague language gets high greenwashing,
# Alto-Mayo-style specific language gets low. Just for pipeline test.
def fake_gw(claims):
    vague_markers = ["world-class", "committed", "best in class", "genuine", "meaningful",
                     "substantial", "high-integrity", "high-impact", "pride", "lasting"]
    probs = []
    per = []
    for c in claims:
        is_vague = any(m in c.lower() for m in vague_markers)
        p = 0.88 if is_vague else 0.22
        probs.append(p)
        per.append({"text": c, "greenwashing_prob": p,
                    "label": "greenwashing" if p >= 0.5 else "credible"})
    avg = sum(probs) / len(probs) * 100 if probs else 0
    return {"aggregate_score": avg, "per_claim": per, "n_claims": len(claims)}


for p in projects:
    bb, _ = fetch_sentinel2_tile(p["lat"], p["lon"], p["date_start"], p["date_end"],
                                  synthetic_forest_bias=p.get("synthetic_forest_bias_before"))
    ba, _ = fetch_sentinel2_tile(p["lat"], p["lon"],
                                  p["date_after_start"], p["date_after_end"],
                                  synthetic_forest_bias=p.get("synthetic_forest_bias_after"))
    sat = compare_tiles(bb, ba, clf=clf)
    gw = fake_gw(p["summary_claims"])
    r = build_report(
        claimed_forest_cover_pct=p.get("claimed_forest_cover_pct"),
        measured_forest_cover_pct=sat["forest_after_pct"],
        forest_change_pct=sat["forest_change_pct"],
        greenwashing_scores=gw,
        n_numeric_claims=0,
        project_name=p["name"],
    )
    print(f"{p['id']:12s} | claimed={p.get('claimed_forest_cover_pct'):>5.1f}% | "
          f"measured={sat['forest_after_pct']:>5.1f}% | "
          f"change={sat['forest_change_pct']:+5.1f}% | "
          f"gw_agg={gw['aggregate_score']:>5.1f} | "
          f"SCORE={r.integrity_score:>5.1f} ({r.verdict})")
