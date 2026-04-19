# Redirect entry point — no streamlit calls here at all
# HF Spaces uses src/demo/app.py directly via README app_file config
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.demo import app  # noqa
