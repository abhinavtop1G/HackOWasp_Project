import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.demo import app  # noqa — set_page_config is the first call inside app.py
