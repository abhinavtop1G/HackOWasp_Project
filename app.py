"""
Root-level entry point for Streamlit Cloud.
Auto-trains fast models on first boot, then runs the demo.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

@st.cache_resource(show_spinner="First-time setup: training forest classifier (< 5 seconds)...")
def _ensure_models_ready():
    from src import config
    if not config.FOREST_CLASSIFIER_PATH.exists():
        from src.satellite.forest_classifier import train
        train()
    return True

_ensure_models_ready()

from src.demo import app  # noqa: F401
