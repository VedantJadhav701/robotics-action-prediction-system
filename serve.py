#!/usr/bin/env python3
"""
🚀 PhysicalAI RoboticsLSTM Production Inference Server
FastAPI application for real-time robot action prediction
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, validator

# Add src to path
sys.path.insert(0, "./src")  # noqa: E402

from model import RoboticsLSTM  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================


class ActionSequence(BaseModel):
    """Input: last 15 action steps"""

    actions: list[list[float]]  # (15, 34)

    @validator("actions")
    def validate_actions(v):
        if len(v) != 15:
            raise ValueError(f"Expected 15 timesteps, got {len(v)}")
        if any(len(a) != 34 for a in v):
            raise ValueError("Expected 34 action dimensions, got mixed")
        return v


class ObservationSequence(BaseModel):
    """Input: last 15 observation steps"""

    observations: list[list[float]]  # (15, 12)

    @validator("observations")
    def validate_obs(v):
        if len(v) != 15:
            raise ValueError(f"Expected 15 timesteps, got {len(v)}")
        if any(len(o) != 12 for o in v):
            raise ValueError("Expected 12 observation dimensions, got mixed")
        return v


class PredictionRequest(BaseModel):
    """Full inference request"""

    action_sequence: ActionSequence
    observation_sequence: ObservationSequence
    return_full_space: bool = True


class PredictionResponse(BaseModel):
    """Inference response"""

    next_action: list[float]  # 34-dim
    prediction_time_ms: float
    version: str
    timestamp: str


class HealthCheck(BaseModel):
    """Health status"""

    status: str
    is_loaded: bool
    api_version: str
    timestamp: str


# ============================================================================
# PRODUCTION INFERENCE ENGINE (from notebook)
# ============================================================================


class ProductionRoboticsInferenceEngine:
    """Production-ready inference engine with all safety checks"""

    def __init__(
        self,
        model_path=None,
        action_mask_path=None,
        norm_stats_path=None,
        device=None,
    ):
        """Initialize production inference engine"""
        # Resolve paths relative to script location (works in Docker)
        base_dir = Path(__file__).parent / "models"
        model_path = model_path or base_dir / "best.pt"
        action_mask_path = action_mask_path or base_dir / "action_mask.npy"
        norm_stats_path = norm_stats_path or base_dir / "normalization_stats.json"

        # Convert to absolute paths
        model_path = Path(model_path).resolve()
        action_mask_path = Path(action_mask_path).resolve()
        norm_stats_path = Path(norm_stats_path).resolve()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load action dimension mask
        self.action_mask = np.load(action_mask_path)
        self.full_action_dim = len(self.action_mask)
        self.reduced_action_dim = int(self.action_mask.sum())
        self.constant_dims = np.where(~self.action_mask)[0]

        # Load normalization statistics
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)

        self.action_min = np.array(norm_stats["action_bounds"]["min"])
        self.action_max = np.array(norm_stats["action_bounds"]["max"])
        self.obs_min = np.array(norm_stats["obs_bounds"]["min"])
        self.obs_max = np.array(norm_stats["obs_bounds"]["max"])
        self.obs_dim = norm_stats["obs_dim"]

        # Precompute ranges
        self.action_range = self.action_max - self.action_min
        self.obs_range = self.obs_max - self.obs_min
        self.action_range = np.where(self.action_range < 1e-8, 1.0, self.action_range)
        self.obs_range = np.where(self.obs_range < 1e-8, 1.0, self.obs_range)

        # Load model
        self.model = RoboticsLSTM(
            action_dim=self.full_action_dim,
            obs_dim=self.obs_dim,
            hidden_dim=128,
            num_layers=3,
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(
            f"✅ Engine Loaded: {self.full_action_dim}-dim action space, {self.obs_dim}-dim observations"
        )

    def _normalize_actions(self, actions):
        return 2.0 * (actions - self.action_min) / self.action_range - 1.0

    def _denormalize_actions(self, actions_norm):
        return 0.5 * (actions_norm + 1.0) * self.action_range + self.action_min

    def _normalize_obs(self, obs):
        return 2.0 * (obs - self.obs_min) / self.obs_range - 1.0

    def predict(self, action_sequence, observation_sequence, return_full_space=True):
        """Predict next action with full pipeline"""

        single_sample = False
        if action_sequence.ndim == 2:
            action_sequence = np.expand_dims(action_sequence, 0)
            observation_sequence = np.expand_dims(observation_sequence, 0)
            single_sample = True

        # NORMALIZE
        action_norm = self._normalize_actions(action_sequence)
        action_norm = np.clip(action_norm, -1.0, 1.0)

        obs_norm = self._normalize_obs(observation_sequence)
        obs_norm = np.clip(obs_norm, -1.0, 1.0)

        # FORWARD PASS
        with torch.no_grad():
            action_t = torch.from_numpy(action_norm).float().to(self.device)
            obs_t = torch.from_numpy(obs_norm).float().to(self.device)
            pred_norm, _ = self.model(action_t, obs_t)

        # CLAMP (CRITICAL)
        pred_norm = torch.clamp(pred_norm, -1.0, 1.0)

        # DENORMALIZE
        pred = pred_norm.cpu().numpy()
        pred = self._denormalize_actions(pred)

        # CONSTANT DIMS = 0
        pred[:, self.constant_dims] = 0.0

        # FILTER OR RETURN FULL SPACE
        if not return_full_space:
            pred = pred[:, self.action_mask]

        if single_sample:
            pred = pred[0]

        return pred


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="PhysicalAI RoboticsLSTM Inference",
    description="Production inference server for robot action prediction",
    version="1.0.0",
)

# CORS for integration with robot control systems
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
engine: ProductionRoboticsInferenceEngine | None = None
MODEL_VERSION = "1.0.0-prod"

# Request/response tracking
stats = {
    "total_requests": 0,
    "successful_predictions": 0,
    "failed_requests": 0,
    "avg_latency_ms": 0.0,
    "latencies": [],
}

# Prometheus metrics
registry = CollectorRegistry()
request_count = Counter(
    "robotics_api_requests_total",
    "Total API requests",
    ["endpoint", "method"],
    registry=registry,
)
request_latency = Histogram(
    "robotics_api_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    registry=registry,
)
prediction_errors = Counter(
    "robotics_api_prediction_errors_total",
    "Total prediction errors",
    ["error_type"],
    registry=registry,
)
model_inference_time = Histogram(
    "robotics_model_inference_seconds",
    "Model inference time in seconds",
    registry=registry,
)
active_requests = Gauge(
    "robotics_api_active_requests", "Currently active requests", registry=registry
)

# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global engine
    try:
        # Auto-detect device (CUDA if available, else CPU)
        engine = ProductionRoboticsInferenceEngine(device=None)
        logger.info("✅ Model loaded successfully on startup")
    except FileNotFoundError as e:
        logger.error(f"⚠️  Model file not found: {e}")
        logger.error("   Ensure model files are included in Docker image")
        logger.error("   Check COPY models/ command in Dockerfile")
        # Don't crash - allow API to start in degraded mode
        engine = None
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        # Allow startup to continue but log the error
        engine = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global engine
    if engine:
        logger.info("🛑 Shutting down inference engine")


# ============================================================================
# ENDPOINTS
# ============================================================================


@app.get("/health", response_model=HealthCheck)
@app.head("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if engine else "unhealthy",
        "is_loaded": engine is not None,
        "api_version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict next action given action and observation history

    Input:
      - action_sequence: (15, 34) normalized action history
      - observation_sequence: (15, 12) observation history

    Output:
      - next_action: (34,) predicted next action
    """

    global engine, stats

    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    stats["total_requests"] += 1
    start_time = time.time()

    # Track active requests
    active_requests.inc()

    try:
        # Increment request counter
        request_count.labels(endpoint="/predict", method="POST").inc()

        # Convert to numpy
        action_array = np.array(request.action_sequence.actions, dtype=np.float32)
        obs_array = np.array(request.observation_sequence.observations, dtype=np.float32)

        # Predict with timing
        inference_start = time.time()
        next_action = engine.predict(
            action_array, obs_array, return_full_space=request.return_full_space
        )
        inference_time = time.time() - inference_start

        latency_ms = (time.time() - start_time) * 1000
        stats["successful_predictions"] += 1
        stats["latencies"].append(latency_ms)

        # Keep only last 1000 latencies
        if len(stats["latencies"]) > 1000:
            stats["latencies"] = stats["latencies"][-1000:]

        stats["avg_latency_ms"] = np.mean(stats["latencies"])

        # Record metrics
        request_latency.labels(endpoint="/predict").observe(latency_ms / 1000.0)
        model_inference_time.observe(inference_time)

        return {
            "next_action": next_action.tolist(),
            "prediction_time_ms": latency_ms,
            "version": MODEL_VERSION,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        stats["failed_requests"] += 1
        prediction_errors.labels(error_type="inference_error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e

    finally:
        # Decrement active requests
        active_requests.dec()


@app.get("/stats")
async def get_stats():
    """Get inference statistics"""
    return {
        "total_requests": stats["total_requests"],
        "successful": stats["successful_predictions"],
        "failed": stats["failed_requests"],
        "avg_latency_ms": stats["avg_latency_ms"],
        "success_rate": stats["successful_predictions"] / max(stats["total_requests"], 1),
    }


@app.get("/")
@app.head("/")
async def root():
    """Root endpoint with API info"""
    return {
        "api": "PhysicalAI RoboticsLSTM Inference",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "stats": "/stats",
            "docs": "/docs",
        },
    }


# ============================================================================
# BATCH PROCESSING (for offline analysis)
# ============================================================================


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""

    action_sequences: list[list[list[float]]]  # (batch, 15, 34)
    observation_sequences: list[list[list[float]]]  # (batch, 15, 12)


@app.post("/predict-batch")
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""

    global engine, stats

    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        action_array = np.array(request.action_sequences, dtype=np.float32)
        obs_array = np.array(request.observation_sequences, dtype=np.float32)

        # Predict all
        next_actions = []
        for i in range(len(action_array)):
            pred = engine.predict(action_array[i], obs_array[i])
            next_actions.append(pred.tolist())

        latency_ms = (time.time() - start_time) * 1000

        return {
            "next_actions": next_actions,
            "batch_size": len(action_array),
            "total_time_ms": latency_ms,
            "avg_time_per_sample_ms": latency_ms / len(action_array),
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return PlainTextResponse(generate_latest(registry))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
