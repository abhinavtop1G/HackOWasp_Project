"""Quick pipeline test without DistilBERT (stubs out the NLP scorer)."""
import sys
sys.path.insert(0, ".")

from src.satellite.forest_classifier import load_classifier
from src.satellite.planetary_computer_client import fetch_sentinel2_tile
from src.satellite.change_detection import compare_tiles, render_comparison_png
from src.fusion.integrity_score import build_report
from src.nlp.claim_extractor import extract_claim_sentences, extract_numeric_claims
from pathlib import Path

# Use synthetic tiles (no network)
bands_before, meta_b = fetch_sentinel2_tile(-16.75, 28.80, "2016-06-01", "2016-09-30")
bands_after, meta_a = fetch_sentinel2_tile(-16.75, 28.80, "2023-06-01", "2023-09-30")
print(f"Before tile shape: {bands_before.shape}, source: {meta_b.get('source')}")
print(f"After tile shape:  {bands_after.shape}, source: {meta_a.get('source')}")

clf = load_classifier()
sat = compare_tiles(bands_before, bands_after, clf=clf)
print(f"Forest before: {sat['forest_before_pct']:.1f}%")
print(f"Forest after:  {sat['forest_after_pct']:.1f}%")
print(f"Change:        {sat['forest_change_pct']:+.1f}%")

# Fake greenwashing output for testing fusion
fake_gw = {
    "aggregate_score": 72.0,
    "n_claims": 5,
    "per_claim": [
        {"text": "We are committed to sustainability.", "greenwashing_prob": 0.95, "label": "greenwashing"},
        {"text": "Our project is best in class.", "greenwashing_prob": 0.89, "label": "greenwashing"},
        {"text": "The project retained 85% forest cover.", "greenwashing_prob": 0.22, "label": "credible"},
    ],
}

report = build_report(
    claimed_forest_cover_pct=85.0,
    measured_forest_cover_pct=sat["forest_after_pct"],
    forest_change_pct=sat["forest_change_pct"],
    greenwashing_scores=fake_gw,
    n_numeric_claims=1,
    project_name="Kariba REDD+",
)

print("\n=== REPORT ===")
print(f"Integrity Score: {report.integrity_score:.1f}")
print(f"Verdict:         {report.verdict}")
print(f"Sat subscore:    {report.satellite_subscore:.1f}")
print(f"NLP subscore:    {report.nlp_subscore:.1f}")
print(f"Cons subscore:   {report.consistency_subscore:.1f}")
print(f"Explanation:     {report.explanation}")

# Test claim extractor on real-ish text
sample_text = """
Our Kariba REDD+ project is committed to world-class forest protection.
Between 2016 and 2023, canopy cover declined from 85% to 51% per Sentinel-2 analysis.
The project covers 785,000 hectares of forest. We delivered 23 million tonnes CO2 of reductions.
"""
sentences = extract_claim_sentences(sample_text)
numerics = extract_numeric_claims(sample_text)
print(f"\n=== CLAIM EXTRACTION ===")
print(f"Found {len(sentences)} claim sentences:")
for s in sentences:
    print(f"  - {s}")
print(f"Found {len(numerics)} numeric claims:")
for n in numerics:
    print(f"  - {n.kind}: {n.value}")

# Test visualization
png_path = Path("/tmp/test_comparison.png")
render_comparison_png(sat, png_path, title="Kariba REDD+ (test)")
print(f"\nVisualization saved to {png_path}: {png_path.exists()}, size={png_path.stat().st_size} bytes")
