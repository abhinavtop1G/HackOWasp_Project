"""
Fetch Sentinel-2 L2A tiles via Microsoft Planetary Computer STAC API.

Caches downloaded tiles as .npz for instant replay in demo conditions.
Falls back to synthetic tiles if the STAC API is unreachable (hackathon WiFi).

Tile format: numpy array shape (4, H, W) = [B02, B03, B04, B08] in S2 L2A DN.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src import config

log = logging.getLogger("pc_client")


def _cache_key(lat: float, lon: float, start: str, end: str) -> str:
    raw = f"{lat:.4f}_{lon:.4f}_{start}_{end}_{config.TILE_SIZE}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    return config.CACHED_TILES_DIR / f"{key}.npz"


def _synthetic_tile(lat: float, lon: float, year: int, forest_bias: float = 0.7) -> np.ndarray:
    """
    Fallback tile used when STAC is unreachable. Realistic reflectance values;
    forest_bias controls what fraction of pixels look forest-like.
    Deterministic for (lat, lon, year) so repeated calls return same tile.
    """
    seed = int(hashlib.md5(f"{lat}_{lon}_{year}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    H = W = config.TILE_SIZE
    forest_mask = rng.random((H, W)) < forest_bias
    bands = np.zeros((4, H, W), dtype=np.float32)

    # Forest pixels: low visible, high NIR
    bands[0][forest_mask] = rng.normal(350, 80, size=forest_mask.sum())
    bands[1][forest_mask] = rng.normal(650, 120, size=forest_mask.sum())
    bands[2][forest_mask] = rng.normal(450, 100, size=forest_mask.sum())
    bands[3][forest_mask] = rng.normal(3500, 600, size=forest_mask.sum())

    # Non-forest pixels: higher visible, moderate NIR
    nf = ~forest_mask
    bands[0][nf] = rng.normal(1100, 300, size=nf.sum())
    bands[1][nf] = rng.normal(1300, 350, size=nf.sum())
    bands[2][nf] = rng.normal(1800, 400, size=nf.sum())
    bands[3][nf] = rng.normal(2100, 450, size=nf.sum())

    return bands.clip(0, 10000).astype(np.float32)


def fetch_sentinel2_tile(
    lat: float,
    lon: float,
    date_start: str,   # "YYYY-MM-DD"
    date_end: str,
    use_cache: bool = True,
    allow_synthetic_fallback: bool = True,
    synthetic_forest_bias: Optional[float] = None,
) -> tuple[np.ndarray, dict]:
    """
    Returns (bands, metadata).
    bands: float32 array shape (4, H, W) = [B02, B03, B04, B08]
    metadata: dict with 'source', 'scene_id', 'acquired', 'cloud_cover' when real.
    """
    key = _cache_key(lat, lon, date_start, date_end)
    path = _cache_path(key)

    # 1. Cache hit
    if use_cache and path.exists():
        data = np.load(path, allow_pickle=True)
        bands = data["bands"]
        meta = data["meta"].item() if "meta" in data.files else {"source": "cache"}
        log.info(f"Cache hit for ({lat:.3f}, {lon:.3f}) {date_start}..{date_end}")
        return bands, meta

    # 2. Try Planetary Computer
    try:
        bands, meta = _fetch_from_planetary_computer(lat, lon, date_start, date_end)
        # Validate tile is not empty/all-zero (happens when GDAL can't stream COGs)
        if bands.mean() < 10.0:
            raise RuntimeError(
                f"Planetary Computer returned an empty tile (mean={bands.mean():.2f}) — "
                "likely a GDAL/network streaming issue. Falling back to synthetic."
            )
        np.savez_compressed(path, bands=bands, meta=np.array(meta, dtype=object))
        log.info(f"Fetched and cached tile from Planetary Computer: {meta.get('scene_id')}")
        return bands, meta
    except Exception as e:
        log.warning(f"Planetary Computer fetch failed: {e}")
        if not allow_synthetic_fallback:
            raise

    # 3. Synthetic fallback — always use the per-project bias if provided
    year = int(date_start[:4])
    bias = synthetic_forest_bias if synthetic_forest_bias is not None else 0.7
    bands = _synthetic_tile(lat, lon, year, forest_bias=bias)
    meta = {"source": "synthetic_fallback", "year": year, "forest_bias": bias}
    log.info(f"Using synthetic tile for ({lat:.3f}, {lon:.3f}) {year} with forest_bias={bias:.2f}")
    return bands, meta


def _fetch_from_planetary_computer(
    lat: float, lon: float, date_start: str, date_end: str
) -> tuple[np.ndarray, dict]:
    """Real STAC fetch. Raises on failure."""
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.windows import Window

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Tiny bounding box around point
    d = 0.025  # ~2.5 km at equator
    bbox = [lon - d, lat - d, lon + d, lat + d]

    search = catalog.search(
        collections=[config.SENTINEL2_COLLECTION],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": config.MAX_CLOUD_COVER}},
        max_items=5,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(f"No Sentinel-2 items for bbox={bbox} {date_start}..{date_end}")

    # Pick the least-cloudy scene
    item = min(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))

    bands_out = []
    for band_name in config.SENTINEL2_BANDS:
        asset = item.assets[band_name]
        with rasterio.open(asset.href) as src:
            # Window centered on (lat, lon)
            row, col = src.index(lon, lat)
            half = config.TILE_SIZE // 2
            window = Window(col - half, row - half, config.TILE_SIZE, config.TILE_SIZE)
            data = src.read(1, window=window, boundless=True, fill_value=0)
            if data.shape != (config.TILE_SIZE, config.TILE_SIZE):
                # Resize if STAC gave us a different-shape chip
                from PIL import Image
                img = Image.fromarray(data)
                img = img.resize((config.TILE_SIZE, config.TILE_SIZE), Image.BILINEAR)
                data = np.array(img)
            bands_out.append(data.astype(np.float32))

    bands = np.stack(bands_out)
    meta = {
        "source": "planetary_computer",
        "scene_id": item.id,
        "acquired": item.properties.get("datetime"),
        "cloud_cover": item.properties.get("eo:cloud_cover"),
    }
    return bands, meta
