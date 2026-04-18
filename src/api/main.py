"""
FastAPI backend for VerifEarth.

Endpoints:
  GET  /health                 -> liveness
  GET  /projects               -> list hero projects
  POST /verify/project         -> verify a hero project by ID
  POST /verify/pdf             -> verify an arbitrary PDF upload (+ optional coords)

OWASP alignment documented per endpoint:
  - LLM10 Unbounded Consumption: slowapi rate limit per IP
  - LLM05 Improper Output Handling: strict Pydantic response models
  - LLM03/ML06 Supply Chain: SHA-256 provenance hash on every report
  - LLM02 Sensitive Info Disclosure: we never echo uploaded PDF bytes
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src import config
from src.fusion.integrity_score import build_report
from src.nlp.claim_extractor import (
    extract_claim_sentences,
    extract_numeric_claims,
    primary_forest_cover_claim,
)
from src.nlp.greenwashing_scorer import GreenwashingScorer
from src.nlp.pdf_extractor import extract_text_from_bytes
from src.satellite.change_detection import compare_tiles
from src.satellite.forest_classifier import load_classifier
from src.satellite.planetary_computer_client import fetch_sentinel2_tile

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="VerifEarth API", version=config.PROVENANCE_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hackathon only; tighten for prod
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return {
        "error": "rate_limit_exceeded",
        "detail": str(exc),
        "owasp_control": "LLM10 Unbounded Consumption",
    }


# ---------- Shared singletons ----------
_scorer: Optional[GreenwashingScorer] = None
_forest_clf = None
_hero_projects: Optional[dict] = None


def get_scorer() -> GreenwashingScorer:
    global _scorer
    if _scorer is None:
        _scorer = GreenwashingScorer()
    return _scorer


def get_forest_clf():
    global _forest_clf
    if _forest_clf is None:
        _forest_clf = load_classifier()
    return _forest_clf


def get_hero_projects() -> dict:
    global _hero_projects
    if _hero_projects is None:
        with open(config.HERO_PROJECTS_PATH) as f:
            _hero_projects = json.load(f)
    return _hero_projects


# ---------- Models ----------
class VerifyProjectRequest(BaseModel):
    project_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: dict


def _provenance_hash(payload: dict) -> str:
    """SHA-256 over (project, timestamp, version, score). OWASP LLM03 defense."""
    key_fields = {
        "project_id": payload.get("project_id"),
        "timestamp": payload.get("timestamp"),
        "version": config.PROVENANCE_VERSION,
        "integrity_score": payload.get("integrity_score"),
    }
    return hashlib.sha256(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()


def _run_full_pipeline(
    *,
    project_name: str,
    project_id: str,
    lat: float,
    lon: float,
    date_start: str,
    date_end: str,
    date_after_start: str,
    date_after_end: str,
    summary_claims: list[str],
    claimed_forest_cover_pct: Optional[float] = None,
    full_text: Optional[str] = None,
    synthetic_bias_before: Optional[float] = None,
    synthetic_bias_after: Optional[float] = None,
) -> dict:
    t0 = time.time()

    # Satellite
    bands_before, meta_before = fetch_sentinel2_tile(
        lat, lon, date_start, date_end,
        synthetic_forest_bias=synthetic_bias_before,
    )
    bands_after, meta_after = fetch_sentinel2_tile(
        lat, lon, date_after_start, date_after_end,
        synthetic_forest_bias=synthetic_bias_after,
    )
    sat_compare = compare_tiles(bands_before, bands_after, clf=get_forest_clf())

    # NLP
    if full_text:
        claim_sentences = extract_claim_sentences(full_text)
        numeric_claims = extract_numeric_claims(full_text)
        primary_claim_pct = primary_forest_cover_claim(numeric_claims) or claimed_forest_cover_pct
    else:
        claim_sentences = summary_claims
        numeric_claims = []
        primary_claim_pct = claimed_forest_cover_pct

    gw = get_scorer().score_claims(claim_sentences)

    # Fusion
    report = build_report(
        claimed_forest_cover_pct=primary_claim_pct,
        measured_forest_cover_pct=sat_compare["forest_after_pct"],
        forest_change_pct=sat_compare["forest_change_pct"],
        greenwashing_scores=gw,
        n_numeric_claims=len(numeric_claims),
        project_name=project_name,
    )

    result = report.to_dict()
    result.update({
        "project_id": project_id,
        "project_name": project_name,
        "timestamp": int(time.time()),
        "satellite_details": {
            "forest_before_pct": sat_compare["forest_before_pct"],
            "forest_after_pct": sat_compare["forest_after_pct"],
            "forest_change_pct": sat_compare["forest_change_pct"],
            "ndvi_before_mean": sat_compare["ndvi_before_mean"],
            "ndvi_after_mean": sat_compare["ndvi_after_mean"],
            "source_before": meta_before,
            "source_after": meta_after,
        },
        "numeric_claims": [
            {"kind": c.kind, "value": c.value, "context": c.context[:140]}
            for c in numeric_claims
        ],
        "processing_ms": int((time.time() - t0) * 1000),
    })
    result["provenance_sha256"] = _provenance_hash(result)
    return result


# ---------- Endpoints ----------
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        version=config.PROVENANCE_VERSION,
        models_loaded={
            "greenwashing": config.GREENWASHING_MODEL_DIR.exists(),
            "forest_classifier": config.FOREST_CLASSIFIER_PATH.exists(),
        },
    )


@app.get("/projects")
def list_projects():
    data = get_hero_projects()
    return {
        "projects": [
            {k: p[k] for k in ("id", "name", "country", "project_type") if k in p}
            for p in data["projects"]
        ]
    }


@app.post("/verify/project")
@limiter.limit(config.API_RATE_LIMIT)
def verify_project(request: Request, body: VerifyProjectRequest):
    data = get_hero_projects()
    match = next((p for p in data["projects"] if p["id"] == body.project_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Project {body.project_id} not found")

    # Check cache
    cache_path = config.CACHED_RESULTS_DIR / f"{body.project_id}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        cached["_cache_hit"] = True
        return cached

    result = _run_full_pipeline(
        project_name=match["name"],
        project_id=match["id"],
        lat=match["lat"],
        lon=match["lon"],
        date_start=match["date_start"],
        date_end=match["date_end"],
        date_after_start=match["date_after_start"],
        date_after_end=match["date_after_end"],
        summary_claims=match["summary_claims"],
        claimed_forest_cover_pct=match.get("claimed_forest_cover_pct"),
        synthetic_bias_before=match.get("synthetic_forest_bias_before"),
        synthetic_bias_after=match.get("synthetic_forest_bias_after"),
    )

    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    result["_cache_hit"] = False
    return result


@app.post("/verify/pdf")
@limiter.limit(config.API_RATE_LIMIT)
async def verify_pdf(
    request: Request,
    pdf: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
    date_start: str = Form("2016-06-01"),
    date_end: str = Form("2016-09-30"),
    date_after_start: str = Form("2023-06-01"),
    date_after_end: str = Form("2023-09-30"),
    project_name: str = Form("Uploaded project"),
):
    # Basic input validation (OWASP LLM05)
    if pdf.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail=f"Expected PDF, got {pdf.content_type}")

    raw = await pdf.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF exceeds 20 MB")

    text = extract_text_from_bytes(raw)
    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from PDF")

    return _run_full_pipeline(
        project_name=project_name,
        project_id=f"upload-{hashlib.md5(raw).hexdigest()[:8]}",
        lat=lat,
        lon=lon,
        date_start=date_start,
        date_end=date_end,
        date_after_start=date_after_start,
        date_after_end=date_after_end,
        summary_claims=[],
        full_text=text,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
