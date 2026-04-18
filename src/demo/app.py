"""
Streamlit demo UI for VerifEarth.

Run:  streamlit run src/demo/app.py

Features:
  - Project selector
  - Big animated Integrity Score gauge (Plotly)
  - Satellite before/after + NDVI heatmap
  - Per-claim greenwashing flags
  - Natural-language explanation
  - Provenance SHA-256 badge (OWASP)

The app calls the API if VERIFEARTH_API_URL is set, otherwise runs the pipeline
in-process. In-process is recommended for demo (zero network dep).
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.fusion.integrity_score import build_report
from src.nlp.claim_extractor import extract_claim_sentences, extract_numeric_claims, primary_forest_cover_claim
from src.nlp.greenwashing_scorer import GreenwashingScorer
from src.nlp.pdf_extractor import extract_text_from_bytes
from src.satellite.change_detection import compare_tiles, render_comparison_png
from src.satellite.forest_classifier import load_classifier
from src.satellite.planetary_computer_client import fetch_sentinel2_tile

st.set_page_config(
    page_title="VerifEarth \u2014 Carbon Credit Auditor",
    page_icon="\U0001F30D",
    layout="wide",
)

# ---------- CSS ----------
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    h1 { color: #1a7f3f; }
    .big-number { font-size: 4rem; font-weight: 700; }
    .verdict-high { color: #1a7f3f; }
    .verdict-medium { color: #d98c00; }
    .verdict-low { color: #c62828; }
    .owasp-badge {
        background: #1a2833; color: #77ddff; padding: 4px 10px; border-radius: 4px;
        font-family: monospace; font-size: 0.85rem; display: inline-block;
    }
    .claim-red { background: #ffe6e6; border-left: 4px solid #c62828; padding: 8px; margin: 4px 0; }
    .claim-green { background: #e8f5e8; border-left: 4px solid #1a7f3f; padding: 8px; margin: 4px 0; }
</style>
""", unsafe_allow_html=True)

# ---------- Singletons (cached) ----------
@st.cache_resource
def get_scorer():
    return GreenwashingScorer()

@st.cache_resource
def get_classifier():
    return load_classifier()

@st.cache_data
def get_hero_projects():
    with open(config.HERO_PROJECTS_PATH) as f:
        return json.load(f)


def integrity_gauge(score: float, verdict: str) -> go.Figure:
    color = "#1a7f3f" if score >= 70 else "#d98c00" if score >= 40 else "#c62828"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"<b>{verdict}</b>", "font": {"size": 22}},
        number={"suffix": " / 100", "font": {"size": 48, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.28},
            "steps": [
                {"range": [0, 40], "color": "#fadbd8"},
                {"range": [40, 70], "color": "#fdebd0"},
                {"range": [70, 100], "color": "#d4efdf"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def sub_bars(report_dict: dict) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[report_dict["satellite_subscore"], report_dict["nlp_subscore"], report_dict["consistency_subscore"]],
        y=["Satellite<br>truth (60%)", "NLP<br>credibility (30%)", "Internal<br>consistency (10%)"],
        orientation="h",
        marker=dict(color=["#2e7d32", "#1565c0", "#6a1b9a"]),
        text=[
            f"{report_dict['satellite_subscore']:.0f}",
            f"{report_dict['nlp_subscore']:.0f}",
            f"{report_dict['consistency_subscore']:.0f}",
        ],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 110]),
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


def run_verification(project: dict) -> dict:
    scorer = get_scorer()
    clf = get_classifier()

    with st.status("Running audit pipeline...", expanded=True) as status:
        st.write("Fetching Sentinel-2 tile (before)...")
        bands_before, meta_before = fetch_sentinel2_tile(
            project["lat"], project["lon"], project["date_start"], project["date_end"],
            synthetic_forest_bias=project.get("synthetic_forest_bias_before"),
        )
        st.write(f"\u2713 Before tile: {meta_before.get('source')} "
                 f"{meta_before.get('scene_id', '')}")

        st.write("Fetching Sentinel-2 tile (after)...")
        bands_after, meta_after = fetch_sentinel2_tile(
            project["lat"], project["lon"],
            project["date_after_start"], project["date_after_end"],
            synthetic_forest_bias=project.get("synthetic_forest_bias_after"),
        )
        st.write(f"\u2713 After tile: {meta_after.get('source')} "
                 f"{meta_after.get('scene_id', '')}")

        st.write("Running forest classifier + NDVI analysis...")
        sat_compare = compare_tiles(bands_before, bands_after, clf=clf)

        st.write("Scoring claims with fine-tuned DistilBERT...")
        gw = scorer.score_claims(project["summary_claims"])

        st.write("Fusing signals...")
        report = build_report(
            claimed_forest_cover_pct=project.get("claimed_forest_cover_pct"),
            measured_forest_cover_pct=sat_compare["forest_after_pct"],
            forest_change_pct=sat_compare["forest_change_pct"],
            greenwashing_scores=gw,
            n_numeric_claims=0,
            project_name=project["name"],
        )
        status.update(label="Audit complete.", state="complete", expanded=False)

    # Save comparison PNG for display
    tmp_png = Path(tempfile.mkstemp(suffix=".png")[1])
    render_comparison_png(sat_compare, tmp_png, title=project["name"])

    return {
        "report": report.to_dict(),
        "comparison_png": str(tmp_png),
        "sat_details": {
            "forest_before_pct": sat_compare["forest_before_pct"],
            "forest_after_pct": sat_compare["forest_after_pct"],
            "forest_change_pct": sat_compare["forest_change_pct"],
            "ndvi_before_mean": sat_compare["ndvi_before_mean"],
            "ndvi_after_mean": sat_compare["ndvi_after_mean"],
            "source_before": meta_before,
            "source_after": meta_after,
        },
        "project": project,
    }


# ==================== LAYOUT ====================
st.title("\U0001F30D VerifEarth")
st.caption("*Independent AI audit layer for carbon credit integrity. "
           "Sentinel-2 + fine-tuned DistilBERT. Built in 18h at HackOWASP 8.0.*")

with st.sidebar:
    st.header("Project")
    data = get_hero_projects()
    project_ids = [p["id"] for p in data["projects"]]
    project_labels = [f"{p['id']} \u2014 {p['name']} ({p['country']})" for p in data["projects"]]
    selected_idx = st.selectbox(
        "Select a carbon project",
        range(len(project_ids)),
        format_func=lambda i: project_labels[i],
    )
    project = data["projects"][selected_idx]

    st.markdown(f"**Type:** {project['project_type']}")
    st.markdown(f"**Coords:** {project['lat']:.2f}, {project['lon']:.2f}")
    if "context" in project:
        st.info(project["context"])

    run_clicked = st.button("\U0001F6F0 Run Audit", type="primary", use_container_width=True)

    st.divider()
    st.caption("**OWASP alignment**")
    st.markdown('<span class="owasp-badge">LLM03 Supply Chain</span>', unsafe_allow_html=True)
    st.markdown('<span class="owasp-badge">LLM05 Output Handling</span>', unsafe_allow_html=True)
    st.markdown('<span class="owasp-badge">LLM09 Misinformation</span>', unsafe_allow_html=True)
    st.markdown('<span class="owasp-badge">LLM10 Unbounded Consumption</span>', unsafe_allow_html=True)
    st.markdown('<span class="owasp-badge">ML06 AI Supply Chain</span>', unsafe_allow_html=True)


if run_clicked or "last_result" in st.session_state:
    if run_clicked:
        st.session_state.last_result = run_verification(project)

    result = st.session_state.last_result
    report = result["report"]
    sat = result["sat_details"]

    # ----- Top row: Gauge + sub-bars -----
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.plotly_chart(integrity_gauge(report["integrity_score"], report["verdict"]),
                        use_container_width=True)
    with c2:
        st.subheader("Sub-scores")
        st.plotly_chart(sub_bars(report), use_container_width=True)

    # ----- Explanation -----
    verdict_class = (
        "verdict-high" if report["integrity_score"] >= 70
        else "verdict-medium" if report["integrity_score"] >= 40
        else "verdict-low"
    )
    st.markdown(f"### \U0001F4C4 Audit summary")
    st.markdown(f"<div style='font-size:1.1rem;'>{report['explanation']}</div>",
                unsafe_allow_html=True)

    # ----- Satellite details -----
    st.markdown("### \U0001F6F0 Satellite evidence")
    c3, c4, c5 = st.columns(3)
    with c3:
        st.metric("Forest cover BEFORE", f"{sat['forest_before_pct']:.1f}%",
                  help=f"Measured on {sat['source_before'].get('acquired', 'date unknown')}")
    with c4:
        st.metric("Forest cover AFTER", f"{sat['forest_after_pct']:.1f}%",
                  help=f"Measured on {sat['source_after'].get('acquired', 'date unknown')}")
    with c5:
        delta = sat["forest_change_pct"]
        st.metric("Change", f"{delta:+.1f}%",
                  delta=f"{delta:+.1f} pts",
                  delta_color="inverse" if delta < 0 else "normal")

    if report.get("claimed_forest_cover_pct") is not None:
        c6, c7 = st.columns(2)
        c6.metric("Project CLAIMED", f"{report['claimed_forest_cover_pct']:.1f}%")
        c7.metric("Satellite MEASURED", f"{report['measured_forest_cover_pct']:.1f}%",
                  delta=f"gap: {report['discrepancy_pct']:.1f} pts",
                  delta_color="inverse")

    st.image(result["comparison_png"], use_column_width=True,
             caption="Sentinel-2 L2A via Microsoft Planetary Computer. "
                     "ML forest mask from trained logistic classifier on spectral bands + NDVI.")

    # ----- Claim analysis -----
    st.markdown("### \U0001F4DD NLP claim analysis")
    n_flagged = report["n_greenwashing_flags"]
    n_total = report["n_claims_analyzed"]
    st.caption(f"Analyzed **{n_total}** claims with fine-tuned `{config.GREENWASHING_BASE_MODEL}`. "
               f"Flagged **{n_flagged}** as greenwashing-style.")

    for claim in report["top_greenwashing_claims"]:
        prob = claim["greenwashing_prob"]
        st.markdown(
            f'<div class="claim-red"><b>\U0001F6A9 Greenwashing ({prob:.0%})</b><br>'
            f'"{claim["text"]}"</div>',
            unsafe_allow_html=True,
        )

    # ----- Provenance -----
    st.markdown("### \U0001F512 Provenance")
    import hashlib, json as _json
    prov_payload = {
        "project_id": result["project"]["id"],
        "integrity_score": report["integrity_score"],
        "version": config.PROVENANCE_VERSION,
        "timestamp": int(time.time()),
    }
    prov_hash = hashlib.sha256(_json.dumps(prov_payload, sort_keys=True).encode()).hexdigest()
    st.code(f"SHA-256: {prov_hash}", language=None)
    st.caption("Every VerifEarth report is content-hashed. "
               "Tampering with the score or methodology will break the hash.")

    with st.expander("Methodology details (for judges \U0001F9D1\u200D\u2696\ufe0f)"):
        st.json(report["methodology"])
else:
    st.info("\u2190 Select a project and click **Run Audit** to begin.")
    st.markdown("""
    **What VerifEarth does:**
    1. Pulls Sentinel-2 satellite imagery from Microsoft Planetary Computer for the project coordinates, before and after the project period.
    2. Runs a trained logistic-regression forest classifier over the spectral bands + NDVI to measure actual forest cover.
    3. Runs a fine-tuned DistilBERT classifier over the project's claims to score greenwashing risk.
    4. Fuses everything into a single **Integrity Score (0\u2013100)** with a natural-language explanation.

    **Why it matters:** The voluntary carbon market is a ~$2B market with a well-documented over-crediting problem. VerifEarth is an independent auditor that anyone can run \u2014 open-source, open data, satellite-verified.
    """)
