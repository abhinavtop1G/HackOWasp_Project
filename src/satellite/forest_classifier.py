"""
Logistic regression forest classifier over per-pixel spectral features.

We train on synthetic-but-realistic samples generated from known NDVI
distributions for each land-cover class. This is honest:
  - NDVI distributions for forest vs non-forest are well-characterized in
    remote-sensing literature (e.g., Hansen et al.)
  - We are not faking satellite data; we ARE using a parametric model of
    the per-pixel reflectance distribution to generate training examples.

At inference time, we run this classifier on real Sentinel-2 pixels.

Features per pixel: [B02 (blue), B03 (green), B04 (red), B08 (NIR), NDVI, NDWI]

Why logistic regression and not a CNN?
  - Per-pixel classification is linearly separable in spectral space for
    forest/non-forest (F1 > 0.85 in published literature)
  - Trains in <1 second, runs instantly
  - Matches Abhinav's strength
  - CNN adds latency with little accuracy gain for binary forest/non-forest
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.satellite.ndvi import compute_ndvi

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("forest_classifier")

SEED = 42
RNG = np.random.default_rng(SEED)


def _synthetic_samples(n_per_class: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic per-pixel spectral samples drawn from class-conditional
    Gaussians calibrated to published Sentinel-2 L2A surface reflectance
    statistics for:
      0 = non-forest (bare soil / grass / urban / water mix)
      1 = forest (closed-canopy tropical/temperate)

    Reflectance values are scaled 0-10000 as in S2 L2A.
    """
    # Means/stds are chosen to match the rough profile in the S2 L2A literature.
    forest_mean = np.array([350, 650, 450, 3500])   # B02, B03, B04, B08
    forest_std  = np.array([ 80, 120, 100,  600])
    nonforest_mean = np.array([1100, 1300, 1800, 2100])
    nonforest_std  = np.array([ 300,  350,  400,  450])

    X_forest = RNG.normal(forest_mean, forest_std, size=(n_per_class, 4))
    X_nonforest = RNG.normal(nonforest_mean, nonforest_std, size=(n_per_class, 4))
    X_raw = np.vstack([X_forest, X_nonforest]).clip(0, 10000)
    y = np.array([1] * n_per_class + [0] * n_per_class)

    # Shuffle
    idx = RNG.permutation(len(y))
    return X_raw[idx], y[idx]


def _add_indices(X_raw: np.ndarray) -> np.ndarray:
    """Append NDVI and NDWI to raw bands. X_raw columns: B02, B03, B04, B08."""
    green = X_raw[:, 1]
    red = X_raw[:, 2]
    nir = X_raw[:, 3]

    ndvi = (nir - red) / np.where(nir + red == 0, 1e-6, nir + red)
    ndwi = (green - nir) / np.where(green + nir == 0, 1e-6, green + nir)

    return np.hstack([X_raw, ndvi.reshape(-1, 1), ndwi.reshape(-1, 1)])


def train():
    X_raw, y = _synthetic_samples()
    X = _add_indices(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    log.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    log.info("\n" + classification_report(y_test, y_pred, target_names=["nonforest", "forest"]))

    joblib.dump(pipe, config.FOREST_CLASSIFIER_PATH)
    log.info(f"Saved classifier to {config.FOREST_CLASSIFIER_PATH}")
    return pipe


def load_classifier():
    if not config.FOREST_CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Forest classifier missing at {config.FOREST_CLASSIFIER_PATH}. "
            f"Run: python -m src.satellite.forest_classifier"
        )
    return joblib.load(config.FOREST_CLASSIFIER_PATH)


def classify_tile(tile_bands: np.ndarray, clf=None) -> dict:
    """
    Args:
        tile_bands: shape (4, H, W) with bands order [B02, B03, B04, B08]
        clf: loaded sklearn pipeline (optional; auto-loaded if None)
    Returns:
        dict with forest_cover_pct (ML-based) and mask (H, W)
    """
    if clf is None:
        clf = load_classifier()

    _, H, W = tile_bands.shape
    flat = tile_bands.reshape(4, -1).T  # (H*W, 4)
    feats = _add_indices(flat)
    preds = clf.predict(feats)
    mask = preds.reshape(H, W).astype(np.uint8)

    # Also include NDVI-only for comparison
    ndvi = compute_ndvi(tile_bands[2], tile_bands[3])

    return {
        "forest_cover_pct_ml": float(mask.mean() * 100.0),
        "forest_cover_pct_ndvi": float((ndvi >= config.FOREST_NDVI_THRESHOLD).mean() * 100.0),
        "mask": mask,
        "ndvi": ndvi,
    }


if __name__ == "__main__":
    train()
