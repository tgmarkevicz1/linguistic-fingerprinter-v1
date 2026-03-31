"""
main.py — FastAPI Application Layer
=====================================
Endpoints:
  POST /enroll              — Register an author with sample texts
  POST /identify            — Attribute an unknown text (top 3 + SHAP)
  GET  /compare/{author_id} — Radar chart: last identified text vs. enrolled author
  POST /compare-texts       — Compare two arbitrary texts directly (score + radar chart)
  POST /retrain             — Upload a zip of new data and fully retrain the model
  GET  /retrain/status      — Poll retraining progress
  GET  /authors             — List enrolled authors
  DEL  /authors/{author_id} — Remove an author
"""

import io
import json
import pickle
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.model_selection import train_test_split

from engine import StylometricEngine
from processors import TextProcessor

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Linguistic Fingerprinting Engine",
    description="Cross-medium authorship attribution via stylometric analysis.",
    version="2.1.0",
)

# Allow the frontend (index.html opened from the filesystem) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

engine = StylometricEngine()
processor = TextProcessor()

# Registry: author_id -> mean feature vector (numpy array)
# Populated at startup from disk, updated by /enroll and /retrain
author_registry: dict[str, np.ndarray] = {}

# The last unknown text's feature vector, stored by /identify so /compare can use it
_last_unknown_vector: Optional[np.ndarray] = None

# Retraining status dict — polled by the frontend via GET /retrain/status
retrain_status: dict = {"running": False, "message": "Idle", "success": None}

MIN_TOKENS = 10

# Cosine similarity threshold for the same-author verdict in /compare-texts.
# 1.0 = identical vectors, 0.0 = completely dissimilar.
# 0.75 is a conservative default — lower it if you're getting too many "different" verdicts.
SAME_AUTHOR_THRESHOLD = 0.75

# Filesystem paths
REGISTRY_SAVE_PATH = Path("models/checkpoints/author_registry.pkl")
RETRAIN_DATA_DIR   = Path("data/retrain_upload")

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class EnrollRequest(BaseModel):
    author_id: str
    texts: list[str]

    @field_validator("texts")
    @classmethod
    def texts_not_empty(cls, v):
        if not v:
            raise ValueError("texts list must not be empty.")
        return v


class IdentifyRequest(BaseModel):
    text: str


class CompareTextsRequest(BaseModel):
    """
    Two texts to compare directly — no enrolled authors needed.
    label_a / label_b are optional display names for the radar chart legend.
    """
    text_a: str
    text_b: str
    label_a: str = "Text A"
    label_b: str = "Text B"


class EnrollResponse(BaseModel):
    author_id: str
    texts_enrolled: int
    message: str


class AuthorMatch(BaseModel):
    author: str
    probability: float


class IdentifyResponse(BaseModel):
    top_matches: list[AuthorMatch]
    shap_features: list[dict]
    token_count: int


class CompareTextsResponse(BaseModel):
    """
    similarity:  cosine similarity of the two feature vectors (0.0–1.0)
    confidence:  same value as a percentage string, e.g. "83.4%"
    verdict:     human-readable same/different conclusion
    radar_chart: Plotly figure JSON for the frontend to render directly
    """
    similarity: float
    confidence: str
    verdict: str
    radar_chart: dict


class RetrainStatusResponse(BaseModel):
    running: bool
    message: str
    success: Optional[bool]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_token_count(text: str) -> int:
    """Raise HTTP 400 if the text has fewer than MIN_TOKENS tokens."""
    count = processor.token_count(text)
    if count < MIN_TOKENS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input text has only {count} token(s). "
                f"A minimum of {MIN_TOKENS} tokens is required for reliable attribution."
            ),
        )
    return count


def _load_engine_if_available() -> bool:
    """Restore engine state from saved checkpoints. Returns True on success."""
    try:
        engine.load()
        return True
    except FileNotFoundError:
        return False


def _build_radar_chart(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    label_a: str,
    label_b: str,
    title: str,
) -> dict:
    """
    Build a Plotly radar chart overlaying two feature vectors.

    Uses the top-8 SHAP features as axes so the chart visualises
    the dimensions that most strongly discriminate between authors —
    not arbitrary feature indices which would be meaningless to look at.

    Returns a plain dict (JSON-serialisable) that Plotly.newPlot() accepts directly.
    """
    top_features = engine.top_shap_features(n=8)

    if not top_features:
        # Graceful fallback if SHAP values are unavailable
        feature_labels = [f"dim_{i}" for i in range(min(8, len(vec_a)))]
        a_vals = vec_a[:8].tolist()
        b_vals = vec_b[:8].tolist()
    else:
        feature_labels = [f["feature"] for f in top_features]
        selected_names = [engine.feature_names[i] for i in engine.selected_feature_indices]

        a_vals, b_vals = [], []
        for label in feature_labels:
            idx = selected_names.index(label) if label in selected_names else None
            a_vals.append(float(vec_a[idx]) if idx is not None else 0.0)
            b_vals.append(float(vec_b[idx]) if idx is not None else 0.0)

    # Min-max normalise both vectors together so axes share a common scale
    all_vals = a_vals + b_vals
    v_min, v_max = min(all_vals), max(all_vals)
    v_range = v_max - v_min if v_max != v_min else 1.0
    norm_a = [(v - v_min) / v_range for v in a_vals]
    norm_b = [(v - v_min) / v_range for v in b_vals]

    # Radar polygons must be closed — repeat the first point at the end
    cats   = feature_labels + [feature_labels[0]]
    norm_a = norm_a + [norm_a[0]]
    norm_b = norm_b + [norm_b[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_a, theta=cats, fill="toself", name=label_a,
        line=dict(color="#4a90d9", width=2),
        fillcolor="rgba(74,144,217,0.2)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=norm_b, theta=cats, fill="toself", name=label_b,
        line=dict(color="#e84393", width=2),
        fillcolor="rgba(232,67,147,0.2)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor="#0d1117"),
        showlegend=True,
        paper_bgcolor="#0d1117",
        font=dict(color="#e0e0e0", family="monospace"),
        title=dict(text=title, font=dict(size=16, color="#4a90d9")),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=60, b=40, l=40, r=40),
    )
    return json.loads(fig.to_json())


def _run_retrain(data_dir: Path) -> None:
    """
    Background task: read .txt files from data_dir, retrain the engine,
    save new checkpoints, and hot-swap the live model without a server restart.

    Called by the /retrain endpoint via FastAPI's BackgroundTasks mechanism.
    The retrain_status dict is updated throughout so the frontend can poll progress.

    data_dir structure (extracted from the uploaded zip):
      data_dir/
        author_a/  text1.txt  text2.txt ...
        author_b/  text1.txt  text2.txt ...
    """
    global retrain_status

    try:
        retrain_status = {"running": True, "message": "Loading uploaded texts...", "success": None}

        # ── Collect texts ─────────────────────────────────────────────────
        # Walk the extracted zip directory, treating each subfolder as one author.
        # We skip files that are too short — they add noise without signal.
        MIN_RETRAIN_TOKENS = 30
        author_texts: dict[str, list[str]] = {}

        for author_dir in sorted(data_dir.iterdir()):
            if not author_dir.is_dir():
                continue
            texts = []
            for txt_file in sorted(author_dir.glob("*.txt")):
                content = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
                if len(content.split()) >= MIN_RETRAIN_TOKENS:
                    texts.append(content)
            if len(texts) >= 2:
                author_texts[author_dir.name] = texts

        if len(author_texts) < 2:
            retrain_status = {
                "running": False,
                "message": "Need at least 2 authors with ≥2 texts each in the zip.",
                "success": False,
            }
            return

        total = sum(len(v) for v in author_texts.values())
        retrain_status["message"] = f"Extracting features for {total} texts..."

        # ── Build training arrays ─────────────────────────────────────────
        # Flatten the author->texts dict into parallel lists.
        # The medium label ("long"/"short") drives adversarial feature selection —
        # it tells the engine which features correlate with text length vs. true style.
        texts_flat, authors_flat, mediums_flat = [], [], []
        for author_id, posts in author_texts.items():
            for text in posts:
                medium = "long" if len(text.split()) > 100 else "short"
                texts_flat.append(text)
                authors_flat.append(author_id)
                mediums_flat.append(medium)

        # Hold out 20% for evaluation (stratified so every author appears in both splits)
        train_texts, _, train_authors, _, train_mediums, _ = train_test_split(
            texts_flat, authors_flat, mediums_flat,
            test_size=0.2, random_state=42, stratify=authors_flat,
        )

        retrain_status["message"] = "Training model (SHAP computation is the slow step)..."

        # ── Retrain ───────────────────────────────────────────────────────
        # We create a brand-new engine rather than re-fitting the existing one.
        # This avoids any state leakage from the previous training run.
        new_engine = StylometricEngine()
        new_engine.fit(
            train_texts,
            train_authors,
            medium_labels=train_mediums,
            top_k_features=200,
        )

        retrain_status["message"] = "Saving checkpoints..."
        new_engine.save()

        # ── Rebuild the author registry ───────────────────────────────────
        # The old registry vectors are stale — they were computed in the previous
        # model's feature space. After retraining, the selected feature indices
        # and scaler parameters have changed, so we must recompute everything.
        new_registry: dict[str, np.ndarray] = {}
        for author_id, posts in author_texts.items():
            masked = [new_engine.processor.mask_pii(t) for t in posts]
            vecs = np.array([new_engine.get_feature_vector(t) for t in masked])
            new_registry[author_id] = vecs.mean(axis=0)

        # Persist the refreshed registry to disk
        REGISTRY_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_SAVE_PATH, "wb") as f:
            pickle.dump(new_registry, f)

        # ── Hot-swap the live engine and registry ─────────────────────────
        # We mutate the module-level `engine` object in place so the already-
        # running FastAPI process immediately picks up the new model.
        # All future requests will use the new weights and feature space.
        engine.__dict__.update(new_engine.__dict__)
        author_registry.clear()
        author_registry.update(new_registry)

        retrain_status = {
            "running": False,
            "message": f"Done. {len(new_registry)} authors enrolled with new model.",
            "success": True,
        }

    except Exception as exc:
        retrain_status = {
            "running": False,
            "message": f"Retrain failed: {str(exc)}",
            "success": False,
        }
    finally:
        # Always clean up the temporary upload directory
        if data_dir.exists():
            shutil.rmtree(data_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event():
    # Restore the trained model from checkpoints written by train.py
    loaded = _load_engine_if_available()
    if loaded:
        print("[API] Pre-trained engine loaded from checkpoints.")
    else:
        print("[API] No checkpoints found — run train.py first.")

    # Restore the author registry so the frontend has authors to work with
    # immediately, without manual re-enrollment after every server restart
    if REGISTRY_SAVE_PATH.exists():
        with open(REGISTRY_SAVE_PATH, "rb") as f:
            saved = pickle.load(f)
        author_registry.update(saved)
        print(f"[API] Loaded {len(saved)} pre-enrolled authors from registry.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(
        "<h2>Linguistic Fingerprinting Engine v2.1</h2>"
        "<p>API running. See <a href='/docs'>Swagger docs</a> or open the frontend.</p>"
    )


@app.post("/enroll", response_model=EnrollResponse)
async def enroll(request: EnrollRequest):
    """
    Register an author by computing and storing their mean fingerprint vector.
    The engine must already be trained (run train.py or /retrain first).
    """
    if not engine._is_fitted:
        raise HTTPException(status_code=503, detail="Engine not trained. Run train.py first.")

    valid_texts = []
    for text in request.texts:
        try:
            _validate_token_count(text)
            valid_texts.append(engine.processor.mask_pii(text))
        except HTTPException:
            continue  # Skip texts that are too short rather than rejecting the whole request

    if not valid_texts:
        raise HTTPException(
            status_code=400,
            detail=f"All provided texts were below the {MIN_TOKENS}-token minimum.",
        )

    vectors = np.array([engine.get_feature_vector(t) for t in valid_texts])
    author_registry[request.author_id] = vectors.mean(axis=0)

    return EnrollResponse(
        author_id=request.author_id,
        texts_enrolled=len(valid_texts),
        message=f"Author '{request.author_id}' enrolled with {len(valid_texts)} text(s).",
    )


@app.post("/identify", response_model=IdentifyResponse)
async def identify(request: IdentifyRequest):
    """
    Attribute an unknown text to the most likely enrolled author(s).
    Stores the text's feature vector so GET /compare/{author_id} can use it.
    """
    if not engine._is_fitted:
        raise HTTPException(status_code=503, detail="Engine not trained. Run train.py first.")

    token_count = _validate_token_count(request.text)
    masked = engine.processor.mask_pii(request.text)

    global _last_unknown_vector
    _last_unknown_vector = engine.get_feature_vector(masked)

    return IdentifyResponse(
        top_matches=[AuthorMatch(**m) for m in engine.top_matches(masked, n=3)],
        shap_features=engine.top_shap_features(n=5),
        token_count=token_count,
    )


@app.get("/compare/{author_id}")
async def compare_with_author(author_id: str):
    """
    Radar chart comparing the last /identify text against an enrolled author's mean profile.
    """
    if _last_unknown_vector is None:
        raise HTTPException(status_code=400, detail="Run /identify first.")
    if author_id not in author_registry:
        raise HTTPException(status_code=404, detail=f"Author '{author_id}' not enrolled.")

    return _build_radar_chart(
        vec_a=_last_unknown_vector,
        vec_b=author_registry[author_id],
        label_a="Unknown Text",
        label_b=f"Author: {author_id}",
        title=f"Stylometric Comparison: Unknown vs. {author_id}",
    )


@app.post("/compare-texts", response_model=CompareTextsResponse)
async def compare_texts(request: CompareTextsRequest):
    """
    Directly compare two texts without needing any enrolled authors.

    How the similarity score works:
      - Both texts are run through the full feature extraction pipeline.
      - We compute the cosine similarity between their feature vectors.
        Cosine similarity measures the *angle* between two vectors in
        high-dimensional feature space — 1.0 means they point in exactly
        the same direction (identical stylometric profile), 0.0 means
        they are completely orthogonal (nothing in common stylistically).
      - The score is compared against SAME_AUTHOR_THRESHOLD (default 0.75)
        to produce a binary verdict.

    This is "authorship verification" — answering "same person?" rather
    than "who wrote this?", which is a harder and often more useful task.
    """
    if not engine._is_fitted:
        raise HTTPException(status_code=503, detail="Engine not trained. Run train.py first.")

    _validate_token_count(request.text_a)
    _validate_token_count(request.text_b)

    # Mask PII so personal names don't inflate similarity artificially
    masked_a = engine.processor.mask_pii(request.text_a)
    masked_b = engine.processor.mask_pii(request.text_b)

    # Extract feature vectors using the trained pipeline
    vec_a = engine.get_feature_vector(masked_a)
    vec_b = engine.get_feature_vector(masked_b)

    # scipy returns cosine *distance* (1 - similarity), so we invert it
    raw_sim = float(1.0 - cosine_distance(vec_a, vec_b))
    similarity = max(0.0, min(1.0, raw_sim))  # clamp for floating-point safety

    verdict = (
        "Likely same author"
        if similarity >= SAME_AUTHOR_THRESHOLD
        else "Likely different authors"
    )

    chart = _build_radar_chart(
        vec_a=vec_a,
        vec_b=vec_b,
        label_a=request.label_a,
        label_b=request.label_b,
        title=f"Stylometric Comparison: {request.label_a} vs. {request.label_b}",
    )

    return CompareTextsResponse(
        similarity=round(similarity, 4),
        confidence=f"{similarity * 100:.1f}%",
        verdict=verdict,
        radar_chart=chart,
    )


@app.post("/retrain")
async def retrain(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a .zip file of new training data and fully retrain the model.

    Expected zip structure — one subfolder per author, each with .txt files:
      my_data.zip
        author_a/
          post1.txt
          post2.txt
        author_b/
          post1.txt
          post2.txt

    Retraining runs in the background so the API stays responsive.
    Poll GET /retrain/status to watch progress.

    On completion, the live model is hot-swapped in memory and the author
    registry is rebuilt — no server restart needed.
    """
    if retrain_status["running"]:
        raise HTTPException(
            status_code=409,
            detail="A retrain is already running. Poll /retrain/status.",
        )

    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload must be a .zip file.")

    contents = await file.read()

    # Extract the zip to a fresh temporary directory
    extract_dir = RETRAIN_DATA_DIR / "current"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as zf:
            # Only extract .txt files — block path traversal (zip-slip) attacks
            for member in zf.infolist():
                if member.filename.endswith(".txt") and ".." not in member.filename:
                    zf.extract(member, extract_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="File is not a valid zip archive.")

    # Queue the actual training as a background task so we can return immediately
    background_tasks.add_task(_run_retrain, extract_dir)

    return {
        "message": "Retraining started in the background.",
        "poll": "GET /retrain/status",
    }


@app.get("/retrain/status", response_model=RetrainStatusResponse)
async def get_retrain_status():
    """
    Poll this to check whether a background retrain is running.
    The frontend polls every 2 seconds and updates the status banner.
    """
    return RetrainStatusResponse(**retrain_status)


@app.get("/authors")
async def list_authors():
    """Return all currently enrolled author IDs."""
    return {"authors": list(author_registry.keys())}


@app.delete("/authors/{author_id}")
async def remove_author(author_id: str):
    """Remove an author from the in-memory registry."""
    if author_id not in author_registry:
        raise HTTPException(status_code=404, detail=f"Author '{author_id}' not found.")
    del author_registry[author_id]
    return {"message": f"Author '{author_id}' removed."}
