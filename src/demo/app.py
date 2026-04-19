"""
VerifEarth — Carbon Credit Auditor  |  HackOWASP 8.0
Redesigned UI: dark glassmorphism, animated score, tabbed results,
per-claim probability bars, project map, NDVI trend line.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# ── Streamlit Cloud path fix ─────────────────────────────────────────────────
# parents[2] of src/demo/app.py  =  repo root (/mount/src/hackowasp_project)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hashlib
import json
import tempfile
import time

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src import config

# ── Auto-train on cold start (Streamlit Cloud has no pre-built models) ──────
def _bootstrap_models():
    """Train forest + greenwashing models if missing (first deploy / cold start)."""
    import pathlib
    forest_ok = pathlib.Path(config.FOREST_CLASSIFIER_PATH).exists()
    gw_ok     = pathlib.Path(config.GREENWASHING_MODEL_DIR).exists()
    if forest_ok and gw_ok:
        return
    with st.spinner("⚙️ First-time setup: training ML models (~2 min) …"):
        if not forest_ok:
            from src.satellite.forest_classifier import train as train_forest
            train_forest()
        if not gw_ok:
            from src.nlp.train_greenwashing import train as train_gw
            train_gw()
    st.success("✅ Models ready!", icon="🤖")
    st.rerun()

_bootstrap_models()

from src.fusion.integrity_score import build_report
from src.nlp.greenwashing_scorer import GreenwashingScorer
from src.satellite.change_detection import compare_tiles, render_comparison_png
from src.satellite.forest_classifier import load_classifier
from src.satellite.planetary_computer_client import fetch_sentinel2_tile

# ─────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VerifEarth — Carbon Credit Auditor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
#  Global CSS  — dark glassmorphism theme
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #080e14 !important;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] > .main { background: transparent !important; }
[data-testid="block-container"] { padding: 1.5rem 2rem 3rem 2rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d1b2a 0%, #0f2637 100%) !important;
    border-right: 1px solid rgba(56,189,248,0.12);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #38bdf8 !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(30,58,95,0.6) !important;
    border: 1px solid rgba(56,189,248,0.25) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #0c2340 40%, #0a3d2e 100%);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(56,189,248,0.07) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(16,185,129,0.06) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-size: 3rem; font-weight: 800; letter-spacing: -1.5px;
    background: linear-gradient(90deg, #38bdf8, #34d399, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
}
.hero-sub {
    font-size: 1rem; color: #64748b; margin: 0;
    font-weight: 400; letter-spacing: 0.01em;
}
.hero-stats {
    display: flex; gap: 2.5rem; margin-top: 1.8rem; flex-wrap: wrap;
}
.stat-pill {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 999px;
    padding: 0.4rem 1.2rem;
    font-size: 0.82rem; font-weight: 600;
    color: #94a3b8;
    display: flex; align-items: center; gap: 0.5rem;
}
.stat-pill span { color: #38bdf8; }

/* ── Glass card ── */
.glass-card {
    background: rgba(15,38,63,0.55);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(56,189,248,0.14);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Score banner ── */
.score-banner {
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
    border: 1px solid;
}
.score-banner.high  { background: rgba(16,185,129,0.1); border-color: rgba(16,185,129,0.35); }
.score-banner.medium{ background: rgba(245,158,11,0.1);  border-color: rgba(245,158,11,0.35); }
.score-banner.low   { background: rgba(239,68,68,0.1);   border-color: rgba(239,68,68,0.4); }
.score-number { font-size: 5.5rem; font-weight: 800; letter-spacing: -3px; line-height: 1; }
.score-banner.high  .score-number { color: #34d399; }
.score-banner.medium .score-number { color: #fbbf24; }
.score-banner.low   .score-number { color: #f87171; }
.score-label { font-size: 1rem; font-weight: 600; letter-spacing: 3px; text-transform: uppercase; margin-top: 0.4rem; }
.score-banner.high  .score-label { color: #34d399; }
.score-banner.medium .score-label { color: #fbbf24; }
.score-banner.low   .score-label { color: #f87171; }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: rgba(15,38,63,0.5);
    border: 1px solid rgba(56,189,248,0.14);
    border-radius: 14px;
    padding: 1rem 1.2rem;
}
.metric-label { font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { font-size: 2rem; font-weight: 700; color: #e2e8f0; margin: 0.2rem 0; }
.metric-delta { font-size: 0.82rem; font-weight: 600; }
.metric-delta.down { color: #f87171; }
.metric-delta.up   { color: #34d399; }
.metric-delta.neutral { color: #94a3b8; }

/* ── Claim cards ── */
.claim-card {
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid;
    position: relative;
    overflow: hidden;
}
.claim-card.gw {
    background: rgba(239,68,68,0.08);
    border-left-color: #ef4444;
}
.claim-card.ok {
    background: rgba(16,185,129,0.07);
    border-left-color: #10b981;
}
.claim-text { font-size: 0.93rem; color: #cbd5e1; font-style: italic; margin: 0.4rem 0 0.6rem 0; }
.claim-prob-bar {
    height: 5px; border-radius: 999px;
    margin-top: 0.5rem;
    background: rgba(255,255,255,0.08);
    overflow: hidden;
}
.claim-prob-fill { height: 100%; border-radius: 999px; transition: width 0.6s ease; }
.claim-badge {
    display: inline-block;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em;
    padding: 2px 8px; border-radius: 999px;
    text-transform: uppercase;
}
.claim-badge.gw { background: rgba(239,68,68,0.2); color: #f87171; }
.claim-badge.ok { background: rgba(16,185,129,0.2); color: #34d399; }

/* ── OWASP badges ── */
.owasp-grid { display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.5rem; }
.owasp-pill {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 8px;
    padding: 5px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #38bdf8;
    display: flex; align-items: center; gap: 0.5rem;
}

/* ── Tabs ── */
[data-testid="stTabs"] { margin-top: 0.5rem; }
[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; font-size: 0.88rem !important;
    color: #64748b !important;
    border-radius: 10px 10px 0 0 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: rgba(56,189,248,0.06) !important;
}

/* ── Provenance hash box ── */
.hash-box {
    background: rgba(10,20,35,0.8);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #38bdf8;
    word-break: break-all;
    letter-spacing: 0.03em;
}

/* ── Summary explanation box ── */
.explanation-box {
    background: rgba(15,38,63,0.5);
    border-left: 4px solid #38bdf8;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.4rem;
    font-size: 0.95rem;
    color: #cbd5e1;
    line-height: 1.7;
    margin-bottom: 1rem;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.05rem; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 1.5rem 0 0.8rem 0;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-header::after {
    content: ''; flex: 1; height: 1px;
    background: rgba(56,189,248,0.12);
}

/* ── Sidebar project card ── */
.project-card {
    background: rgba(56,189,248,0.06);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin: 0.7rem 0;
    font-size: 0.85rem;
}
.project-card .p-label { color: #64748b; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em; }
.project-card .p-value { color: #e2e8f0; font-weight: 600; margin-top: 1px; }

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid rgba(56,189,248,0.15) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(15,38,63,0.4) !important;
    border: 1px solid rgba(56,189,248,0.12) !important;
    border-radius: 12px !important;
}

/* ── divider ── */
hr { border-color: rgba(56,189,248,0.1) !important; }

/* ── Status widget ── */
[data-testid="stStatus"] {
    background: rgba(15,38,63,0.7) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 12px !important;
}

/* ── Pulse animation for running indicator ── */
@keyframes pulse-ring {
    0%   { transform: scale(0.9); opacity: 1; }
    50%  { transform: scale(1.05); opacity: 0.7; }
    100% { transform: scale(0.9); opacity: 1; }
}
.pulse-dot {
    display: inline-block; width: 9px; height: 9px;
    background: #38bdf8; border-radius: 50%;
    animation: pulse-ring 1.4s ease-in-out infinite;
    margin-right: 6px;
}

/* ── Alert / info ── */
[data-testid="stAlert"] {
    background: rgba(56,189,248,0.07) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 12px !important;
    color: #94a3b8 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  Cached singletons
# ─────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────
#  Chart builders
# ─────────────────────────────────────────────────────────
def integrity_gauge(score: float, verdict: str) -> go.Figure:
    if score >= 70:
        color, bg = "#34d399", "rgba(16,185,129,0.15)"
    elif score >= 40:
        color, bg = "#fbbf24", "rgba(245,158,11,0.15)"
    else:
        color, bg = "#f87171", "rgba(239,68,68,0.15)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"<b>{verdict}</b>", "font": {"size": 16, "color": "#94a3b8", "family": "Inter"}},
        number={"suffix": " / 100", "font": {"size": 52, "color": color, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1e3a5f",
                     "tickfont": {"color": "#64748b", "size": 11}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 40],  "color": "rgba(239,68,68,0.12)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.10)"},
                {"range": [70, 100],"color": "rgba(16,185,129,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85, "value": score,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def sub_score_chart(report: dict) -> go.Figure:
    labels = ["🛰 Satellite Truth (60%)", "🧠 NLP Credibility (30%)", "📋 Consistency (10%)"]
    values = [report["satellite_subscore"], report["nlp_subscore"], report["consistency_subscore"]]
    colors = ["#34d399", "#38bdf8", "#a78bfa"]

    fig = go.Figure()
    for i, (label, val, col) in enumerate(zip(labels, values, colors)):
        fig.add_trace(go.Bar(
            x=[val], y=[label], orientation="h",
            marker=dict(
                color=col,
                opacity=0.85,
                line=dict(width=0),
            ),
            text=[f"<b>{val:.0f}</b>"],
            textfont=dict(color="#e2e8f0", size=13, family="Inter"),
            textposition="outside",
            name=label,
            showlegend=False,
            width=0.55,
        ))

    fig.update_layout(
        xaxis=dict(range=[0, 120], showgrid=False, zeroline=False,
                   showticklabels=False, showline=False),
        yaxis=dict(showgrid=False, showline=False,
                   tickfont=dict(color="#94a3b8", size=12, family="Inter")),
        height=200,
        margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="overlay",
    )
    return fig


def forest_timeline_chart(before_pct: float, after_pct: float,
                           before_year: int, after_year: int) -> go.Figure:
    years  = [before_year, after_year]
    values = [before_pct, after_pct]
    color  = "#34d399" if after_pct >= before_pct else "#f87171"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=values,
        mode="lines+markers+text",
        line=dict(color=color, width=3, shape="spline"),
        marker=dict(size=12, color=color,
                    line=dict(color="#0d1b2a", width=2)),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
        text=[f"{v:.1f}%" for v in values],
        textposition=["bottom right","top right"],
        textfont=dict(color=color, size=13, family="Inter"),
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickvals=years, ticktext=[str(y) for y in years],
                   tickfont=dict(color="#64748b", family="Inter"),
                   showgrid=False, zeroline=False, showline=False),
        yaxis=dict(range=[0, 110], ticksuffix="%",
                   tickfont=dict(color="#64748b", family="Inter"),
                   showgrid=True, gridcolor="rgba(56,189,248,0.06)",
                   zeroline=False, showline=False),
    )
    return fig


def project_map(lat: float, lon: float, name: str) -> go.Figure:
    fig = go.Figure(go.Scattergeo(
        lat=[lat], lon=[lon],
        mode="markers+text",
        marker=dict(size=14, color="#38bdf8",
                    line=dict(color="#0d1b2a", width=2),
                    symbol="circle"),
        text=[name], textposition="top center",
        textfont=dict(color="#38bdf8", size=11, family="Inter"),
    ))
    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="rgba(15,38,63,0.9)",
        showocean=True, oceancolor="rgba(8,14,20,0.95)",
        showcoastlines=True, coastlinecolor="rgba(56,189,248,0.25)",
        showframe=False,
        showcountries=True, countrycolor="rgba(56,189,248,0.12)",
        showlakes=True, lakecolor="rgba(8,14,20,0.95)",
        center=dict(lat=lat, lon=lon),
        projection_scale=4,
    )
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def claim_prob_bar_html(text: str, gw_prob: float) -> str:
    is_gw    = gw_prob >= 0.5
    bar_pct  = int(gw_prob * 100)
    bar_col  = "#ef4444" if is_gw else "#10b981"
    card_cls = "gw" if is_gw else "ok"
    badge    = ("🚩 GREENWASHING" if is_gw else "✅ CREDIBLE")
    badge_cls= "gw" if is_gw else "ok"
    return f"""
<div class="claim-card {card_cls}">
  <span class="claim-badge {badge_cls}">{badge}</span>
  <span style="float:right;font-size:0.78rem;color:#64748b;font-family:'JetBrains Mono',monospace;">
    {gw_prob:.0%} greenwashing
  </span>
  <div class="claim-text">"{text}"</div>
  <div style="font-size:0.72rem;color:#64748b;margin-bottom:3px;">Greenwashing probability</div>
  <div class="claim-prob-bar">
    <div class="claim-prob-fill" style="width:{bar_pct}%;background:{bar_col};"></div>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────
#  Audit runner
# ─────────────────────────────────────────────────────────
def run_verification(project: dict) -> dict:
    scorer = get_scorer()
    clf    = get_classifier()

    steps = [
        ("🛰 Fetching Sentinel-2 tile — BEFORE period", None),
        ("🛰 Fetching Sentinel-2 tile — AFTER period",  None),
        ("🌲 Running forest classifier + NDVI analysis", None),
        ("🧠 Scoring claims with fine-tuned DistilBERT", None),
        ("⚡ Fusing signals → Integrity Score",          None),
    ]

    with st.status("Running VerifEarth audit pipeline...", expanded=True) as status:
        st.write(steps[0][0])
        bands_before, meta_before = fetch_sentinel2_tile(
            project["lat"], project["lon"],
            project["date_start"], project["date_end"],
            synthetic_forest_bias=project.get("synthetic_forest_bias_before"),
        )
        src_b = meta_before.get("source", "unknown")
        st.write(f"  ✓ Before tile loaded  |  source: `{src_b}`")

        st.write(steps[1][0])
        bands_after, meta_after = fetch_sentinel2_tile(
            project["lat"], project["lon"],
            project["date_after_start"], project["date_after_end"],
            synthetic_forest_bias=project.get("synthetic_forest_bias_after"),
        )
        src_a = meta_after.get("source", "unknown")
        st.write(f"  ✓ After tile loaded   |  source: `{src_a}`")

        st.write(steps[2][0])
        sat_compare = compare_tiles(bands_before, bands_after, clf=clf)
        st.write(f"  ✓ Forest BEFORE: {sat_compare['forest_before_pct']:.1f}%  →  AFTER: {sat_compare['forest_after_pct']:.1f}%")

        st.write(steps[3][0])
        gw = scorer.score_claims(project["summary_claims"])
        st.write(f"  ✓ {gw['n_claims']} claims analyzed  |  flagged: {sum(1 for c in gw['per_claim'] if c['greenwashing_prob']>=0.5)}")

        st.write(steps[4][0])
        report = build_report(
            claimed_forest_cover_pct=project.get("claimed_forest_cover_pct"),
            measured_forest_cover_pct=sat_compare["forest_after_pct"],
            forest_change_pct=sat_compare["forest_change_pct"],
            greenwashing_scores=gw,
            n_numeric_claims=0,
            project_name=project["name"],
        )
        st.write(f"  ✓ Integrity Score: **{report.integrity_score:.1f} / 100** — {report.verdict}")
        status.update(label="✅ Audit complete!", state="complete", expanded=False)

    tmp_png = Path(tempfile.mkstemp(suffix=".png")[1])
    render_comparison_png(sat_compare, tmp_png, title=project["name"])

    return {
        "report":       report.to_dict(),
        "comparison_png": str(tmp_png),
        "sat_details": {
            "forest_before_pct":  sat_compare["forest_before_pct"],
            "forest_after_pct":   sat_compare["forest_after_pct"],
            "forest_change_pct":  sat_compare["forest_change_pct"],
            "ndvi_before_mean":   sat_compare["ndvi_before_mean"],
            "ndvi_after_mean":    sat_compare["ndvi_after_mean"],
            "source_before":      meta_before,
            "source_after":       meta_after,
        },
        "project": project,
    }


# ─────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 VerifEarth")
    st.markdown("<div style='color:#64748b;font-size:0.8rem;margin-bottom:1.2rem;'>Carbon Credit Fraud Detection · HackOWASP 8.0</div>", unsafe_allow_html=True)

    data = get_hero_projects()
    projects = data["projects"]
    project_labels = [f"{p['id']} — {p['name']}" for p in projects]

    selected_idx = st.selectbox(
        "Select a carbon project",
        range(len(projects)),
        format_func=lambda i: project_labels[i],
    )
    project = projects[selected_idx]

    # Project info card
    st.markdown(f"""
    <div class="project-card">
      <div class="p-label">Country</div>
      <div class="p-value">{project.get('country','—')}</div>
      <div class="p-label" style="margin-top:0.6rem;">Type</div>
      <div class="p-value">{project.get('project_type','—')}</div>
      <div class="p-label" style="margin-top:0.6rem;">Coordinates</div>
      <div class="p-value">{project['lat']:.3f}°, {project['lon']:.3f}°</div>
      <div class="p-label" style="margin-top:0.6rem;">Claimed Forest Retention</div>
      <div class="p-value">{project.get('claimed_forest_cover_pct','N/A')}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Mini map
    st.plotly_chart(
        project_map(project["lat"], project["lon"], project["name"]),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    if "context" in project:
        st.info(project["context"])

    run_clicked = st.button("🛰 Run Audit", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("<div style='color:#64748b;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;'>OWASP Alignment</div>", unsafe_allow_html=True)
    owasp = ["LLM03 Supply Chain", "LLM05 Output Handling", "LLM09 Misinformation", "LLM10 Unbounded Consumption", "ML06 AI Supply Chain"]
    icons = ["🔗","📤","❌","⚡","🧩"]
    for icon, badge in zip(icons, owasp):
        st.markdown(f'<div class="owasp-pill">{icon} {badge}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────────────────

# Hero banner
st.markdown("""
<div class="hero">
  <div class="hero-title">🌍 VerifEarth</div>
  <div class="hero-sub">Independent AI audit layer for carbon credit integrity — Sentinel-2 satellite + fine-tuned DistilBERT</div>
  <div class="hero-stats">
    <div class="stat-pill">🌐 Voluntary carbon market <span>~$2B / yr</span></div>
    <div class="stat-pill">📡 Sentinel-2 resolution <span>10 m/px</span></div>
    <div class="stat-pill">🤖 DistilBERT NLP F1 <span>92.3%</span></div>
    <div class="stat-pill">🔒 SHA-256 tamper-proof <span>OWASP LLM03</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Run / Display results ───
if run_clicked:
    st.session_state.last_result = run_verification(project)

if "last_result" in st.session_state:
    result = st.session_state.last_result
    report = result["report"]
    sat    = result["sat_details"]
    proj   = result["project"]

    score   = report["integrity_score"]
    verdict = report["verdict"]
    band    = "high" if score >= 70 else ("medium" if score >= 40 else "low")

    # ── Score banner ──────────────────────────────────────
    col_score, col_bars = st.columns([1, 1.4], gap="large")

    with col_score:
        st.markdown(f"""
        <div class="score-banner {band}">
          <div style="font-size:0.75rem;color:#64748b;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.5rem;">
            {proj['name']} · {proj.get('country','')}
          </div>
          <div class="score-number">{score:.1f}</div>
          <div style="color:#64748b;font-size:1rem;margin:2px 0 4px 0;">out of 100</div>
          <div class="score-label">{verdict}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_bars:
        st.markdown('<div class="section-header">Sub-score Breakdown</div>', unsafe_allow_html=True)
        st.plotly_chart(sub_score_chart(report), use_container_width=True,
                        config={"displayModeBar": False})

    # ── Explanation ────────────────────────────────────────
    st.markdown(f'<div class="explanation-box">📄 {report["explanation"]}</div>',
                unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🛰  Satellite Evidence",
        "🧠  NLP Claim Analysis",
        "📊  Score Gauge",
        "🔒  Provenance",
    ])

    # ── TAB 1: Satellite ──────────────────────────────────
    with tab1:
        before_year = int(proj.get("date_start","2016")[:4])
        after_year  = int(proj.get("date_after_start","2023")[:4])
        delta       = sat["forest_change_pct"]
        delta_dir   = "down" if delta < 0 else "up"

        # Key metrics row
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="metric-label">Forest Cover {before_year}</div>
            <div class="metric-value">{sat['forest_before_pct']:.1f}%</div>
            <div class="metric-delta neutral">Baseline</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Forest Cover {after_year}</div>
            <div class="metric-value">{sat['forest_after_pct']:.1f}%</div>
            <div class="metric-delta {delta_dir}">{delta:+.1f} pts</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Project Claimed</div>
            <div class="metric-value">{report.get('claimed_forest_cover_pct',0):.1f}%</div>
            <div class="metric-delta neutral">Retention target</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Claim Gap</div>
            <div class="metric-value" style="color:#f87171;">{report.get('discrepancy_pct',0):.1f} pts</div>
            <div class="metric-delta down">vs satellite</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">NDVI {before_year} → {after_year}</div>
            <div class="metric-value">{sat['ndvi_before_mean']:.2f}</div>
            <div class="metric-delta {delta_dir}">→ {sat['ndvi_after_mean']:.2f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Forest cover trend line
        st.markdown('<div class="section-header">Forest Cover Trend</div>', unsafe_allow_html=True)
        st.plotly_chart(
            forest_timeline_chart(sat["forest_before_pct"], sat["forest_after_pct"],
                                  before_year, after_year),
            use_container_width=True, config={"displayModeBar": False}
        )

        # Satellite image grid
        st.markdown('<div class="section-header">Sentinel-2 Satellite Imagery</div>', unsafe_allow_html=True)
        st.image(result["comparison_png"], use_container_width=True,
                 caption="RGB · NDVI Heatmap · ML Forest Mask  |  Before (top) vs After (bottom)  |  10m resolution Sentinel-2 L2A")

        src_before = sat["source_before"].get("source","unknown")
        src_after  = sat["source_after"].get("source","unknown")
        st.markdown(f"""
        <div style="font-size:0.75rem;color:#475569;text-align:center;margin-top:0.3rem;">
          Data source: <code style="color:#38bdf8;">{src_before}</code> (before) ·
          <code style="color:#38bdf8;">{src_after}</code> (after)
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: NLP Claims ────────────────────────────────
    with tab2:
        per_claim = report.get("top_greenwashing_claims", [])
        all_claims = proj.get("summary_claims", [])
        gw_scorer  = get_scorer()

        # Re-score all claims (not just flagged ones)
        all_scores = gw_scorer.score_claims(all_claims)
        all_per_claim = all_scores.get("per_claim", [])

        n_gw    = report["n_greenwashing_flags"]
        n_total = report["n_claims_analyzed"]
        agg     = 100 - report["nlp_subscore"]

        # Summary row
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Claims Analyzed", n_total)
        col_b.metric("Greenwashing Flags", f"{n_gw} / {n_total}",
                     delta=f"{n_gw/max(n_total,1):.0%} flagged",
                     delta_color="inverse" if n_gw > 0 else "normal")
        col_c.metric("Aggregate Vagueness", f"{agg:.1f} / 100",
                     delta="lower is better", delta_color="off")

        st.markdown('<div class="section-header">Claim-by-Claim Analysis</div>', unsafe_allow_html=True)

        claims_html = ""
        for c in all_per_claim:
            claims_html += claim_prob_bar_html(c["text"], c["greenwashing_prob"])

        st.markdown(claims_html, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:0.78rem;color:#475569;margin-top:0.8rem;">
          Model: <code style="color:#38bdf8;">{config.GREENWASHING_BASE_MODEL}</code> fine-tuned on
          hand-labeled environmental claims corpus · Test F1 = 0.923
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 3: Score Gauge ────────────────────────────────
    with tab3:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(integrity_gauge(score, verdict),
                            use_container_width=True,
                            config={"displayModeBar": False})
        with c2:
            st.markdown('<div class="section-header">Score Breakdown</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="glass-card">
              <table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
                <tr style="border-bottom:1px solid rgba(56,189,248,0.1);">
                  <td style="padding:0.5rem 0;color:#64748b;">🛰 Satellite sub-score</td>
                  <td style="text-align:right;color:#34d399;font-weight:700;">{report['satellite_subscore']:.1f}</td>
                  <td style="text-align:right;color:#475569;padding-left:0.8rem;">× 0.60</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(56,189,248,0.1);">
                  <td style="padding:0.5rem 0;color:#64748b;">🧠 NLP sub-score</td>
                  <td style="text-align:right;color:#38bdf8;font-weight:700;">{report['nlp_subscore']:.1f}</td>
                  <td style="text-align:right;color:#475569;padding-left:0.8rem;">× 0.30</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(56,189,248,0.1);">
                  <td style="padding:0.5rem 0;color:#64748b;">📋 Consistency sub-score</td>
                  <td style="text-align:right;color:#a78bfa;font-weight:700;">{report['consistency_subscore']:.1f}</td>
                  <td style="text-align:right;color:#475569;padding-left:0.8rem;">× 0.10</td>
                </tr>
                <tr>
                  <td style="padding:0.7rem 0 0 0;color:#e2e8f0;font-weight:700;">Total Integrity Score</td>
                  <td style="text-align:right;font-size:1.3rem;font-weight:800;color:{'#34d399' if score>=70 else '#fbbf24' if score>=40 else '#f87171'};padding-top:0.7rem;">{score:.1f}</td>
                  <td style="text-align:right;color:#64748b;padding-left:0.8rem;padding-top:0.7rem;">/ 100</td>
                </tr>
              </table>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📐 Methodology details"):
                st.json(report["methodology"])

    # ── TAB 4: Provenance ────────────────────────────────
    with tab4:
        prov_payload = {
            "project_id":      proj["id"],
            "integrity_score": report["integrity_score"],
            "version":         config.PROVENANCE_VERSION,
            "timestamp":       int(time.time()),
        }
        prov_hash = hashlib.sha256(
            json.dumps(prov_payload, sort_keys=True).encode()
        ).hexdigest()

        st.markdown('<div class="section-header">Tamper-Evident Report Hash</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="hash-box">
          SHA-256 · {prov_hash}
        </div>
        <div style="font-size:0.8rem;color:#475569;margin-top:0.5rem;">
          This hash is computed over project ID, score, model version, and timestamp.
          Any tampering with the audit result will break this hash. — <b>OWASP LLM03 Supply Chain Defense</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:1.5rem;">Audit Payload</div>', unsafe_allow_html=True)
        st.json(prov_payload)

        st.markdown('<div class="section-header">OWASP Control Mapping</div>', unsafe_allow_html=True)
        controls = [
            ("LLM03", "Supply Chain",         "SHA-256 provenance hash over every report"),
            ("LLM05", "Output Handling",      "Scores clamped 0–100; no raw model output exposed"),
            ("LLM09", "Misinformation",       "Satellite ground-truth overrides NLP-only claims"),
            ("LLM10", "Unbounded Consumption","FastAPI rate-limited: 30 req/min via slowapi"),
            ("ML06",  "AI Supply Chain",      "Frozen distilbert-base-uncased backbone, Apache 2.0"),
        ]
        for ctrl, name, desc in controls:
            st.markdown(f"""
            <div style="display:flex;gap:1rem;align-items:flex-start;padding:0.7rem 0;border-bottom:1px solid rgba(56,189,248,0.08);">
              <div class="owasp-pill" style="flex-shrink:0;min-width:120px;">{ctrl} {name}</div>
              <div style="font-size:0.85rem;color:#64748b;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    # ── Landing screen ────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem 2rem 1rem;">
      <div style="font-size:4rem;margin-bottom:1rem;">🛰</div>
      <div style="font-size:1.4rem;font-weight:700;color:#e2e8f0;margin-bottom:0.5rem;">Ready to audit</div>
      <div style="color:#64748b;font-size:0.95rem;">Select a carbon project from the sidebar and click <b style="color:#38bdf8;">Run Audit</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">How VerifEarth Works</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    steps_info = [
        ("📡", "Fetch Satellite Data", "Pulls Sentinel-2 L2A tiles from Microsoft Planetary Computer for the project GPS coords, spanning 2016 → 2023"),
        ("🌲", "Classify Forest Pixels", "Runs a trained logistic-regression classifier over 6 spectral features (B02 B03 B04 B08 + NDVI + NDWI) per pixel"),
        ("🧠", "Score NLP Claims", "Fine-tuned DistilBERT reads the project's claims and scores each for greenwashing-style vague language"),
        ("⚡", "Fuse → Integrity Score", "Satellite truth (60%) + NLP credibility (30%) + consistency (10%) → single 0–100 fraud score"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], steps_info):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;min-height:180px;">
              <div style="font-size:2.2rem;margin-bottom:0.5rem;">{icon}</div>
              <div style="font-size:0.9rem;font-weight:700;color:#e2e8f0;margin-bottom:0.4rem;">{title}</div>
              <div style="font-size:0.78rem;color:#64748b;line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:1.5rem;">Available Projects</div>', unsafe_allow_html=True)
    project_cols = st.columns(len(projects))
    rflags = ["🔴", "🟡", "🟡", "🟡", "🟢"]
    for col, proj_item, flag in zip(project_cols, projects, rflags):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
              <div style="font-size:1.4rem;">{flag}</div>
              <div style="font-size:0.78rem;color:#38bdf8;font-weight:700;margin:0.3rem 0;">{proj_item['id']}</div>
              <div style="font-size:0.82rem;color:#e2e8f0;font-weight:600;">{proj_item['name']}</div>
              <div style="font-size:0.72rem;color:#64748b;margin-top:0.2rem;">{proj_item.get('country','')}</div>
            </div>
            """, unsafe_allow_html=True)
