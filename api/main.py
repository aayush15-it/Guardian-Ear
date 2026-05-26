"""
Guardian Ear — FastAPI Service.

Provides REST endpoints for:
- Audio file prediction (``POST /api/v1/predict``)
- Health check (``GET /api/v1/health``)
- System status (``GET /api/v1/status``)

Launch with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Resolve project root for src.* imports ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.threat_engine.tracker import TemporalPatternTracker
from src.threat_engine.rules import ThreatAssessor, get_sound_mode, get_mode_description
from src.features.audio_pipeline import AudioFeatureExtractor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────────
# Pydantic response schemas
# ─────────────────────────────────────────────────
class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    uptime_seconds: float = Field(..., description="Seconds since startup")
    model_loaded: bool = Field(..., description="Whether the model is loaded")


class PredictionResponse(BaseModel):
    """Prediction endpoint response."""
    sound_class: str = Field(..., description="Predicted sound class")
    confidence: float = Field(..., description="Confidence percentage (0-100)")
    threat_score: float = Field(..., description="Threat score (0-100)")
    threat_level: str = Field(..., description="CRITICAL / HIGH / MEDIUM / LOW / SAFE")
    sound_mode: str = Field(..., description="ALERT / ASSISTIVE / NEUTRAL")
    description: str = Field(..., description="Human-readable description")
    pattern_label: str = Field(..., description="Temporal pattern label")
    pattern_score: float = Field(..., description="Pattern intensity (0-1)")
    alert_raised: bool = Field(..., description="Whether the alert threshold was exceeded")


class StatusResponse(BaseModel):
    """System status response."""
    model_loaded: bool
    sample_rate: int
    class_names: List[str]
    location: str
    confidence_threshold: float
    active_patterns: Dict[str, Any]


# ─────────────────────────────────────────────────
# Application state
# ─────────────────────────────────────────────────
_start_time: float = 0.0
_model = None
_extractor: Optional[AudioFeatureExtractor] = None
_assessor: Optional[ThreatAssessor] = None
_tracker: Optional[TemporalPatternTracker] = None
_X_min: Optional[float] = None
_X_max: Optional[float] = None
_config: Dict[str, Any] = {}

CLASS_NAMES: List[str] = [
    'air_conditioner', 'car_horn', 'children_playing',
    'dog_bark', 'drilling', 'engine_idling',
    'gun_shot', 'jackhammer', 'siren', 'street_music',
]
SAMPLE_RATE = 22050
DURATION = 3
CONFIDENCE_THRESHOLD = 0.70
LOCATION = 'unknown'


# ─────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────
app = FastAPI(
    title="Guardian Ear API",
    description="AI-Based Acoustic Threat Detection Service",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event() -> None:
    """Load model and initialise components at startup."""
    global _start_time, _model, _extractor, _assessor, _tracker
    global _X_min, _X_max, _config

    _start_time = time.time()
    logger.info("Guardian Ear API starting up...")

    # Load config if available
    config_path = os.path.join(str(_PROJECT_ROOT), 'configs', 'config.yaml')
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                _config = yaml.safe_load(f) or {}
        except ImportError:
            logger.warning("PyYAML not installed — using defaults")

    # Components
    _tracker = TemporalPatternTracker()
    _assessor = ThreatAssessor(_config)
    _extractor = AudioFeatureExtractor(_config)

    # Load normalization
    model_dir = _config.get('paths', {}).get('model_dir', 'model')
    x_min_path = os.path.join(model_dir, 'X_min.npy')
    x_max_path = os.path.join(model_dir, 'X_max.npy')
    if os.path.exists(x_min_path) and os.path.exists(x_max_path):
        _X_min = float(np.load(x_min_path)[0])
        _X_max = float(np.load(x_max_path)[0])

    # Load model
    model_path = os.path.join(model_dir, 'guardian_ear_model.h5')
    if not os.path.exists(model_path):
        fallback_path = os.path.join(model_dir, 'best_model.h5')
        if os.path.exists(fallback_path):
            model_path = fallback_path

    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            _model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded: %s", model_path)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
    else:
        logger.warning("Model not found at %s", os.path.join(model_dir, 'guardian_ear_model.h5'))


# ─────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────
@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health and uptime."""
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - _start_time, 2),
        model_loaded=_model is not None,
    )


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Accept a .wav audio file and return prediction with threat assessment.

    Args:
        file: Uploaded .wav file.

    Returns:
        PredictionResponse with classification and threat details.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.filename and not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files accepted")

    # Save to temp file
    tmp_path = None
    try:
        import librosa

        contents = await file.read()
        tmp_dir = os.path.join(str(_PROJECT_ROOT), '.tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, f"upload_{time.time_ns()}.wav")

        with open(tmp_path, 'wb') as f:
            f.write(contents)

        # Load audio
        audio, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, duration=DURATION)

        # Extract features
        features = _extractor.extract_features(audio, sr)
        if _X_min is not None and _X_max is not None:
            features = (features - _X_min) / (_X_max - _X_min + 1e-8)
        features = features[np.newaxis, ..., np.newaxis]

        # Predict
        preds = _model.predict(features, verbose=0)[0]
        class_id = int(np.argmax(preds))
        confidence = float(preds[class_id])
        class_name = CLASS_NAMES[class_id]

        # Generate alert
        alert = _assessor.generate_alert(
            sound_class=class_name,
            confidence=confidence,
            location=LOCATION,
            tracker=_tracker,
        )

        return PredictionResponse(
            sound_class=alert.sound_class,
            confidence=alert.confidence,
            threat_score=alert.threat_score,
            threat_level=alert.threat_level,
            sound_mode=alert.sound_mode,
            description=alert.description,
            pattern_label=alert.pattern_label,
            pattern_score=alert.pattern_score,
            alert_raised=alert.alert_raised,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.get("/api/v1/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    """Return current system status and active patterns."""
    active = {}
    if _tracker:
        patterns = _tracker.get_all_active_patterns()
        active = {
            cls: {
                'label': p.pattern_label,
                'score': p.pattern_score,
                'count': p.detection_count,
            }
            for cls, p in patterns.items()
        }

    return StatusResponse(
        model_loaded=_model is not None,
        sample_rate=SAMPLE_RATE,
        class_names=CLASS_NAMES,
        location=LOCATION,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        active_patterns=active,
    )
