import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# set_page_config must be the ONLY call and must be FIRST
# It lives here and NOWHERE else in the codebase
st.set_page_config(
    page_title="VerifEarth — Carbon Credit Auditor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.demo import app  # noqa
