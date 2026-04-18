"""
Central configuration for VerifEarth.
All paths, model names, thresholds, and fusion weights live here so we can
tune without hunting through files.
"""
from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
CACHED_TILES_DIR = DATA_DIR / "cached_tiles"
CACHED_RESULTS_DIR = DATA_DIR / "cached_results"

HERO_PROJECTS_PATH = DATA_DIR / "hero_projects.json"
TRAINING_CLAIMS_PATH = DATA_DIR / "training_claims.json"

# Ensure dirs exist at import
for _d in (MODELS_DIR, CACHED_TILES_DIR, CACHED_RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- NLP ----------
# DistilBERT is small (66M), fine-tunes in minutes on T4, and is the standard
# transfer-learning baseline. We don't use ClimateBERT directly because
# distilbert is 3x smaller and trains faster for a hackathon.
GREENWASHING_BASE_MODEL = "distilbert-base-uncased"
GREENWASHING_MODEL_DIR = MODELS_DIR / "greenwashing_distilbert"
GREENWASHING_MAX_LEN = 128
GREENWASHING_BATCH_SIZE = 16
GREENWASHING_EPOCHS = 4
GREENWASHING_LR = 2e-5

# Claim extraction: a sentence is a "claim" if it contains at least one of these
CLAIM_KEYWORDS = [
    "forest", "carbon", "emission", "sequester", "hectare", "tonne",
    "sustainable", "biodiversity", "deforestation", "conservation",
    "green", "climate", "renewable", "offset", "credit", "REDD",
    "reforest", "afforest", "baseline", "additionality", "leakage",
    "CO2", "CO\u2082", "reduction", "removal", "verified", "certified",
]

# ---------- Satellite ----------
SENTINEL2_COLLECTION = "sentinel-2-l2a"
SENTINEL2_BANDS = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR
# NDVI threshold above which a pixel is considered "forest".
# Hansen et al. and subsequent forest-mapping literature use 0.4-0.6.
FOREST_NDVI_THRESHOLD = 0.5
# Tile size in pixels (each pixel is 10m at Sentinel-2 L2A 10m bands)
# 512x512 = ~5km x 5km = enough for a small carbon project
TILE_SIZE = 512
# Max cloud cover to accept for a scene
MAX_CLOUD_COVER = 20

FOREST_CLASSIFIER_PATH = MODELS_DIR / "forest_classifier.joblib"

# ---------- Fusion weights ----------
# These turn three signals into a single 0-100 Integrity Score.
# Chosen so satellite ground truth dominates, with NLP as a credibility modifier.
FUSION_WEIGHT_SATELLITE = 0.60   # How much the satellite truth disagrees with claims
FUSION_WEIGHT_NLP = 0.30         # How vague/greenwashy the claims are
FUSION_WEIGHT_CONSISTENCY = 0.10 # Internal consistency of the PDF

# ---------- API ----------
API_RATE_LIMIT = "30/minute"  # OWASP LLM10 defense
PROVENANCE_VERSION = "verifearth-v1.0.0-mvp"
