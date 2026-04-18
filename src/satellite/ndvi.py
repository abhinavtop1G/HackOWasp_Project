"""
NDVI (Normalized Difference Vegetation Index) and derived forest-cover estimation.

NDVI = (NIR - Red) / (NIR + Red)

Ranges from -1 to +1:
  - water / bare rock / snow: < 0.1
  - bare soil / sparse grass: 0.1 - 0.3
  - shrubs, grassland:        0.3 - 0.5
  - dense vegetation, forest: 0.5 - 0.9

For Sentinel-2 L2A, Red = B04 and NIR = B08 (both at 10m resolution).
We treat pixels with NDVI >= FOREST_NDVI_THRESHOLD as forest.

This is the "classical remote sensing" baseline. The forest_classifier.py module
trains a small logistic regression on top of NDVI + raw bands for a more
nuanced decision, but this NDVI function is what the basic pipeline uses.
"""
from __future__ import annotations

import numpy as np

from src import config


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Return NDVI array in [-1, 1]. Handles division-by-zero safely."""
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    denom = nir + red
    denom[denom == 0] = 1e-6
    ndvi = (nir - red) / denom
    return np.clip(ndvi, -1.0, 1.0)


def forest_mask_from_ndvi(ndvi: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Binary mask: 1 where forest, 0 otherwise."""
    thr = threshold if threshold is not None else config.FOREST_NDVI_THRESHOLD
    return (ndvi >= thr).astype(np.uint8)


def forest_cover_percent(ndvi: np.ndarray, threshold: float | None = None) -> float:
    """Percentage of pixels classified as forest."""
    mask = forest_mask_from_ndvi(ndvi, threshold)
    return float(mask.mean() * 100.0)


def ndvi_summary(red: np.ndarray, nir: np.ndarray) -> dict:
    """Full diagnostic summary for a tile."""
    ndvi = compute_ndvi(red, nir)
    mask = forest_mask_from_ndvi(ndvi)
    return {
        "ndvi_mean": float(ndvi.mean()),
        "ndvi_median": float(np.median(ndvi)),
        "ndvi_std": float(ndvi.std()),
        "forest_cover_pct": float(mask.mean() * 100.0),
        "tile_shape": list(ndvi.shape),
    }
