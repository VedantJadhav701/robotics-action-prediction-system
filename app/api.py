"""
FastAPI Production Inference Server
PhysicalAI Robotics Action Prediction System
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.model import RoboticsLSTM  # noqa: E402
from src.monitoring import MetricsCollector, PerformanceMonitor  # noqa: E402

# ============================================================================
# Setup
# ============================================================================
app = FastAPI(
    title="PhysicalAI Robotics Action Prediction",
    description="Production inference engine for robot action prediction",
    version="1.0.0",
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
metrics = MetricsCollector()
perf_monitor = PerformanceMonitor()


# ============================================================================
# Models
# ============================================================================
class PredictionRequest(BaseModel):
    action_sequence: list  # (15, 34)
    observation_sequence: list  # (15, 12)
    return_full_space: bool = True


class PredictionResponse(BaseModel):
    prediction: list
    confidence: float
    inference_time_ms: float
    timestamp: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    is_loaded: bool
    device: str
    version: str


# ============================================================================
# Inference Engine (Production-Ready)
# ============================================================================
class ProductionInferenceEngine:
    """Production-grade inference with validation"""

    def __init__(
        self,
        model_path="./models/best.pt",
        action_mask_path="./models/action_mask.npy",
        norm_stats_path="./models/normalization_stats.json",
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load mask
        self.action_mask = np.load(action_mask_path)
        self.full_action_dim = len(self.action_mask)
        self.constant_dims = np.where(~self.action_mask)[0]

        # Load stats
        with open(norm_stats_path) as f:
            stats = json.load(f)

        self.action_min = np.array(stats["action_bounds"]["min"])
        self.action_max = np.array(stats["action_bounds"]["max"])
        self.obs_min = np.array(stats["obs_bounds"]["min"])
        self.obs_max = np.array(stats["obs_bounds"]["max"])
        self.obs_dim = stats["obs_dim"]

        # Ranges
        self.action_range = self.action_max - self.action_min
        self.obs_range = self.obs_max - self.obs_min
        self.action_range = np.where(self.action_range < 1e-8, 1.0, self.action_range)
        self.obs_range = np.where(self.obs_range < 1e-8, 1.0, self.obs_range)

        # Model
        self.model = RoboticsLSTM(
            action_dim=self.full_action_dim,
            obs_dim=self.obs_dim,
            hidden_dim=128,
            num_layers=3,
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info("✅ Production engine initialized")

    def predict(self, action_seq, obs_seq):
        """Make prediction with full validation"""

        # Validate input
        action_seq = np.array(action_seq)
        obs_seq = np.array(obs_seq)

        if action_seq.ndim == 2:
            action_seq = np.expand_dims(action_seq, 0)
            obs_seq = np.expand_dims(obs_seq, 0)
            single = True
        else:
            single = False

        # Check NaN/Inf
        if np.isnan(action_seq).any() or np.isinf(action_seq).any():
            raise ValueError("Invalid action sequence (NaN/Inf)")
        if np.isnan(obs_seq).any() or np.isinf(obs_seq).any():
            raise ValueError("Invalid observation sequence (NaN/Inf)")

        # Normalize
        action_norm = 2.0 * (action_seq - self.action_min) / self.action_range - 1.0
        action_norm = np.clip(action_norm, -1.0, 1.0)

        obs_norm = 2.0 * (obs_seq - self.obs_min) / self.obs_range - 1.0
        obs_norm = np.clip(obs_norm, -1.0, 1.0)

        # Forward pass
        with torch.no_grad():
            action_t = torch.from_numpy(action_norm).float().to(self.device)
            obs_t = torch.from_numpy(obs_norm).float().to(self.device)
            pred_norm, _ = self.model(action_t, obs_t)
            pred_norm = torch.clamp(pred_norm, -1.0, 1.0)

        # Denormalize
        pred = pred_norm.cpu().numpy()
        pred = 0.5 * (pred + 1.0) * self.action_range + self.action_min

        # Constant dims to zero
        pred[:, self.constant_dims] = 0.0

        if single:
            pred = pred[0]

        return pred


# ============================================================================
# Initialize Engine
# ============================================================================
engine = None


def init_engine():
    global engine
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        engine = ProductionInferenceEngine(device=device)
        logger.info("✅ Engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize engine: {e}")
        raise


# ============================================================================
# Routes
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    init_engine()
    logger.info("🚀 API started")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if engine else "initializing",
        "is_loaded": engine is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make prediction"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    import time
    import uuid

    request_id = str(uuid.uuid4())
    start = time.time()

    try:
        # Predict
        pred = engine.predict(request.action_sequence, request.observation_sequence)

        inference_time = (time.time() - start) * 1000

        # Log metrics
        background_tasks.add_task(
            metrics.log_prediction,
            request_id=request_id,
            inference_time_ms=inference_time,
            success=True,
        )

        return {
            "prediction": pred.tolist(),
            "confidence": float(np.mean(np.abs(pred))),
            "inference_time_ms": inference_time,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        background_tasks.add_task(
            metrics.log_prediction,
            request_id=request_id,
            inference_time_ms=(time.time() - start) * 1000,
            success=False,
        )
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    return metrics.get_summary()


@app.get("/docs")
async def docs():
    """API documentation"""
    return {"docs": "See /docs for Swagger UI"}
    return {
        "title": "PhysicalAI Robotics Action Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Make action prediction",
            "GET /health": "Health check",
            "GET /metrics": "Performance metrics",
            "GET /docs": "This documentation",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
