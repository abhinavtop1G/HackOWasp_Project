"""
Before/after satellite comparison and visualization.

Given two tiles from the same location at different dates, computes forest-cover
change and renders a side-by-side PNG with NDVI heatmaps for the demo.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.satellite.forest_classifier import classify_tile
from src.satellite.ndvi import compute_ndvi

log = logging.getLogger("change_detection")


def _to_rgb(bands: np.ndarray) -> np.ndarray:
    """
    Convert (4, H, W) S2 bands to RGB for display.
    Bands order: [B02 blue, B03 green, B04 red, B08 NIR].
    Returns (H, W, 3) in [0, 1].
    """
    rgb = np.stack([bands[2], bands[1], bands[0]], axis=-1)  # R, G, B
    # Simple percentile stretch for display (S2 values roughly 0-4000 for land)
    lo = np.percentile(rgb, 2)
    hi = np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    return rgb


def compare_tiles(bands_before: np.ndarray, bands_after: np.ndarray, clf=None) -> dict:
    r_before = classify_tile(bands_before, clf=clf)
    r_after = classify_tile(bands_after, clf=clf)

    delta_ml = r_after["forest_cover_pct_ml"] - r_before["forest_cover_pct_ml"]
    delta_ndvi = r_after["forest_cover_pct_ndvi"] - r_before["forest_cover_pct_ndvi"]

    return {
        "forest_before_pct": r_before["forest_cover_pct_ml"],
        "forest_after_pct": r_after["forest_cover_pct_ml"],
        "forest_change_pct": delta_ml,
        "forest_before_pct_ndvi": r_before["forest_cover_pct_ndvi"],
        "forest_after_pct_ndvi": r_after["forest_cover_pct_ndvi"],
        "forest_change_pct_ndvi": delta_ndvi,
        "ndvi_before_mean": float(r_before["ndvi"].mean()),
        "ndvi_after_mean": float(r_after["ndvi"].mean()),
        "_internal": {
            "bands_before": bands_before,
            "bands_after": bands_after,
            "ndvi_before": r_before["ndvi"],
            "ndvi_after": r_after["ndvi"],
            "mask_before": r_before["mask"],
            "mask_after": r_after["mask"],
        },
    }


def render_comparison_png(comparison: dict, out_path: Path, title: str = "") -> Path:
    """Write a 2x3 grid: RGB before, RGB after, NDVI before, NDVI after, mask before, mask after."""
    internal = comparison["_internal"]
    rgb_before = _to_rgb(internal["bands_before"])
    rgb_after = _to_rgb(internal["bands_after"])
    ndvi_before = internal["ndvi_before"]
    ndvi_after = internal["ndvi_after"]
    mask_before = internal["mask_before"]
    mask_after = internal["mask_after"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(rgb_before)
    axes[0, 0].set_title(f"RGB Before\n(Forest: {comparison['forest_before_pct']:.1f}%)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(ndvi_before, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
    axes[0, 1].set_title(f"NDVI Before (mean={comparison['ndvi_before_mean']:.2f})")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mask_before, cmap="Greens", vmin=0, vmax=1)
    axes[0, 2].set_title("Forest Mask Before (ML)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(rgb_after)
    axes[1, 0].set_title(f"RGB After\n(Forest: {comparison['forest_after_pct']:.1f}%)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(ndvi_after, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
    axes[1, 1].set_title(f"NDVI After (mean={comparison['ndvi_after_mean']:.2f})")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(mask_after, cmap="Greens", vmin=0, vmax=1)
    axes[1, 2].set_title("Forest Mask After (ML)")
    axes[1, 2].axis("off")

    delta = comparison["forest_change_pct"]
    sign = "+" if delta >= 0 else ""
    suptitle = title or "Satellite Forest Cover Comparison"
    fig.suptitle(f"{suptitle}  |  \u0394 Forest = {sign}{delta:.1f}%", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Wrote comparison PNG to {out_path}")
    return out_path
