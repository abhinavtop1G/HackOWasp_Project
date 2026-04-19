"""
HuggingFace Spaces + Streamlit Cloud entry point.
HF Spaces looks for app.py by default.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

@st.cache_resource(show_spinner="Setting up models (first run only, ~5 seconds)...")
def _ensure_models_ready():
    from src import config
    if not config.FOREST_CLASSIFIER_PATH.exists():
        from src.satellite.forest_classifier import train
        train()
    return True

_ensure_models_ready()

from src.demo import app  # noqa
