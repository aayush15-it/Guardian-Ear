"""
Guardian Ear — FastAPI Service v2.

Provides REST endpoints for:
- Audio file prediction (``POST /api/v1/predict``)
- Health check (``GET /api/v1/health``)
- System status (``GET /api/v1/status``)
- Telegram test alert (``POST /api/v1/alert/test``)
- Send emergency alert (``POST /api/v1/alert/send``)

Launch with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Changes from v1:
  - Replaced deprecated @app.on_event('startup') with lifespan context manager
  - Fixed model_dir config key (was 'paths.model_dir', correct is 'paths.model')
  - Replaced CORS allow_origins=['*'] with env-configurable origins
  - Added /api/v1/alert/test and /api/v1/alert/send endpoints
  - Added API key authentication middleware (optional, env-configured)
  - Dependency injection via fastapi.Depends instead of global mutable state
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
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
# Security — optional API key (set GUARDIAN_API_KEY env var to enable)
# ─────────────────────────────────────────────────
_API_KEY = os.environ.get("GUARDIAN_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-Guardian-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(_api_key_header)) -> bool:
    """Validate API key if one is configured. Skip validation if no key set."""
    if not _API_KEY:
        return True  # API key not configured — open access (dev mode)
    if api_key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return True


# ─────────────────────────────────────────────────
# Application state (wrapped in a class to avoid globals)
# ─────────────────────────────────────────────────
class AppState:
    """Holds all application-level singletons."""
    start_time: float = 0.0
    model = None
    extractor: Optional[AudioFeatureExtractor] = None
    assessor: Optional[ThreatAssessor] = None
    tracker: Optional[TemporalPatternTracker] = None
    X_min: Optional[float] = None
    X_max: Optional[float] = None
    config: Dict[str, Any] = {}


_state = AppState()

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
# Pydantic response schemas
# ─────────────────────────────────────────────────
class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    uptime_seconds: float = Field(..., description="Seconds since startup")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    version: str = Field(default="2.0.0")


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
    is_known: bool = Field(default=True, description="False if open-set rejection applied")


class StatusResponse(BaseModel):
    """System status response."""
    model_loaded: bool
    sample_rate: int
    class_names: List[str]
    location: str
    confidence_threshold: float
    active_patterns: Dict[str, Any]


class AlertRequest(BaseModel):
    """Manual alert dispatch request."""
    sound_class: str = Field(..., description="Sound class to alert on")
    confidence: float = Field(..., description="Confidence 0-1")
    threat_score: float = Field(..., description="Threat score 0-100")
    threat_level: str = Field(..., description="Threat level string")
    location: str = Field(default="unknown")
    action_type: str = Field(default="Manual Test Alert")
    token: Optional[str] = Field(default=None, description="Override Telegram token")
    chat_id: Optional[str] = Field(default=None, description="Override Telegram chat_id")


class AlertResponse(BaseModel):
    """Alert dispatch response."""
    success: bool
    message: str


# ─────────────────────────────────────────────────
# Lifespan (replaces deprecated @app.on_event)
# ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and initialise components at startup."""
    _state.start_time = time.time()
    logger.info("Guardian Ear API v2 starting up...")

    # Load config
    config_path = _PROJECT_ROOT / 'configs' / 'config.yaml'
    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r') as f:
                _state.config = yaml.safe_load(f) or {}
        except ImportError:
            logger.warning("PyYAML not installed — using defaults")

    # Components
    _state.tracker = TemporalPatternTracker()
    _state.assessor = ThreatAssessor(_state.config)
    _state.extractor = AudioFeatureExtractor(_state.config)

    # FIX: correct key is 'paths.model', not 'paths.model_dir'
    model_dir = _state.config.get('paths', {}).get('model', 'model')

    # Load normalization
    x_min_path = _PROJECT_ROOT / model_dir / 'X_min.npy'
    x_max_path = _PROJECT_ROOT / model_dir / 'X_max.npy'
    if x_min_path.exists() and x_max_path.exists():
        _state.X_min = float(np.load(x_min_path)[0])
        _state.X_max = float(np.load(x_max_path)[0])
        logger.info("Normalization loaded: min=%.2f, max=%.2f", _state.X_min, _state.X_max)

    # Load model — try guardian_ear_model.h5 then best_model.h5
    for model_name in ('guardian_ear_model.h5', 'best_model.h5'):
        model_path = _PROJECT_ROOT / model_dir / model_name
        if model_path.exists():
            try:
                import tensorflow as tf
                _state.model = tf.keras.models.load_model(str(model_path))
                logger.info("Model loaded: %s", model_path)
                break
            except Exception as exc:
                logger.error("Failed to load model %s: %s", model_path, exc)

    if _state.model is None:
        logger.warning("No model found in %s — prediction endpoint will return 503", model_dir)

    yield  # Application runs

    # Shutdown
    logger.info("Guardian Ear API shutting down")


# ─────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────
# CORS origins — controlled via environment variable for security
# Production: set GUARDIAN_CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
# Development: leave unset to allow all localhost origins only
_CORS_ORIGINS_ENV = os.environ.get("GUARDIAN_CORS_ORIGINS", "")
_CORS_ORIGINS = (
    [o.strip() for o in _CORS_ORIGINS_ENV.split(",") if o.strip()]
    if _CORS_ORIGINS_ENV
    else [
        "http://localhost",
        "http://localhost:8501",
        "http://localhost:3000",
        "http://127.0.0.1:8501",
    ]
)

app = FastAPI(
    title="Guardian Ear API",
    description="AI-Based Acoustic Threat Detection Service",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────
@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health and uptime."""
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - _state.start_time, 2),
        model_loaded=_state.model is not None,
        version="2.0.0",
    )


@app.post("/api/v1/predict", response_model=PredictionResponse,
          dependencies=[Depends(verify_api_key)])
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Accept a .wav audio file and return prediction with threat assessment.

    Args:
        file: Uploaded .wav file.

    Returns:
        PredictionResponse with classification and threat details.
    """
    if _state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.filename and not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files accepted")

    tmp_path = None
    try:
        import librosa

        contents = await file.read()
        tmp_dir = _PROJECT_ROOT / '.tmp'
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / f"upload_{time.time_ns()}.wav"

        tmp_path.write_bytes(contents)

        # Load audio
        audio, sr = librosa.load(str(tmp_path), sr=SAMPLE_RATE, duration=DURATION)

        # Extract features
        features = _state.extractor.extract_features(audio, sr)
        if _state.X_min is not None and _state.X_max is not None:
            features = (features - _state.X_min) / (_state.X_max - _state.X_min + 1e-8)
        features = features[np.newaxis, ..., np.newaxis]

        # Predict
        preds = _state.model.predict(features, verbose=0)[0]

        # Open-set detection
        is_known = True
        try:
            from src.inference.open_set import OpenSetClassifier
            osc = OpenSetClassifier()
            result = osc.classify(preds)
            is_known = result['is_known']
            if not is_known:
                return PredictionResponse(
                    sound_class='unknown',
                    confidence=round(float(np.max(preds)) * 100, 1),
                    threat_score=0.0,
                    threat_level='SAFE',
                    sound_mode='NEUTRAL',
                    description='Unknown sound — does not match any trained class',
                    pattern_label='BRIEF_NORMAL',
                    pattern_score=0.0,
                    alert_raised=False,
                    is_known=False,
                )
        except ImportError:
            pass  # Open-set module not available

        class_id = int(np.argmax(preds))
        confidence = float(preds[class_id])
        class_name = CLASS_NAMES[class_id]

        # Generate alert
        alert = _state.assessor.generate_alert(
            sound_class=class_name,
            confidence=confidence,
            location=LOCATION,
            tracker=_state.tracker,
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
            is_known=is_known,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


@app.get("/api/v1/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    """Return current system status and active patterns."""
    active = {}
    if _state.tracker:
        patterns = _state.tracker.get_all_active_patterns()
        active = {
            cls: {
                'label': p.pattern_label,
                'score': p.pattern_score,
                'count': p.detection_count,
            }
            for cls, p in patterns.items()
        }

    return StatusResponse(
        model_loaded=_state.model is not None,
        sample_rate=SAMPLE_RATE,
        class_names=CLASS_NAMES,
        location=LOCATION,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        active_patterns=active,
    )


@app.post("/api/v1/alert/test", response_model=AlertResponse,
          dependencies=[Depends(verify_api_key)])
async def test_alert() -> AlertResponse:
    """Send a test Telegram alert to verify integration."""
    try:
        from src.notifications.telegram_service import TelegramAlertService
        token = _state.config.get('telegram', {}).get('token', '')
        chat_id = _state.config.get('telegram', {}).get('chat_id', '')
        service = TelegramAlertService(token=token, chat_id=chat_id)

        if not service.is_configured():
            return AlertResponse(
                success=False,
                message="Telegram not configured. Set GUARDIAN_TELEGRAM_TOKEN and GUARDIAN_TELEGRAM_CHAT_ID."
            )

        ok = service.send_test_alert()
        return AlertResponse(
            success=ok,
            message="Test alert sent successfully." if ok else "Failed to send test alert. Check your token and chat_id."
        )
    except ImportError:
        return AlertResponse(success=False, message="Telegram service module not available.")
    except Exception as exc:
        logger.exception("Test alert failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/alert/send", response_model=AlertResponse,
          dependencies=[Depends(verify_api_key)])
async def send_alert(req: AlertRequest) -> AlertResponse:
    """Send an emergency alert via Telegram.

    Args:
        req: AlertRequest with sound details and optional token override.

    Returns:
        AlertResponse indicating success or failure.
    """
    try:
        from src.notifications.telegram_service import TelegramAlertService

        token = req.token or _state.config.get('telegram', {}).get('token', '')
        chat_id = req.chat_id or _state.config.get('telegram', {}).get('chat_id', '')
        service = TelegramAlertService(token=token, chat_id=chat_id)

        if not service.is_configured():
            return AlertResponse(
                success=False,
                message="Telegram not configured."
            )

        ok = service.send_alert(
            sound_class=req.sound_class,
            confidence=req.confidence * 100,
            threat_level=req.threat_level,
            threat_score=req.threat_score,
            location=req.location,
            action_type=req.action_type,
        )
        return AlertResponse(
            success=ok,
            message="Alert dispatched." if ok else "Failed to dispatch alert."
        )
    except ImportError:
        return AlertResponse(success=False, message="Telegram service module not available.")
    except Exception as exc:
        logger.exception("Alert dispatch failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
