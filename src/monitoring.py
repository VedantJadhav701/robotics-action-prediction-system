"""
Monitoring and Metrics Collection
"""

import logging
import json
from datetime import datetime
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect inference metrics"""

    def __init__(self, save_dir='./logs/metrics'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = []

    def log_prediction(self, request_id, inference_time_ms, success):
        """Log prediction metrics"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "inference_time_ms": inference_time_ms,
            "success": success
        }
        self.metrics.append(metric)

        # Save periodically
        if len(self.metrics) % 100 == 0:
            self._save_metrics()

    def _save_metrics(self):
        """Save metrics to file"""
        timestamp = datetime.now().strftime('%Y%m%d')
        path = self.save_dir / f"metrics_{timestamp}.json"
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Saved {len(self.metrics)} metrics")

    def get_summary(self):
        """Get metrics summary"""
        if not self.metrics:
            return {"message": "No metrics yet"}

        times = [m['inference_time_ms'] for m in self.metrics]
        successes = [m['success'] for m in self.metrics]

        return {
            "total_requests": len(self.metrics),
            "success_rate": np.mean(successes),
            "inference_time_ms": {
                "mean": np.mean(times),
                "median": np.median(times),
                "min": np.min(times),
                "max": np.max(times),
                "std": np.std(times)
            }
        }


class PerformanceMonitor:
    """Monitor model performance"""

    def __init__(self):
        self.predictions = []
        self.targets = []

    def add_batch(self, predictions, targets):
        """Add batch for monitoring"""
        self.predictions.extend(predictions)
        self.targets.extend(targets)

    def compute_metrics(self):
        """Compute performance metrics"""
        if not self.predictions:
            return {}

        pred_array = np.array(self.predictions)
        target_array = np.array(self.targets)

        mae = np.mean(np.abs(pred_array - target_array))
        mse = np.mean((pred_array - target_array) ** 2)
        rmse = np.sqrt(mse)

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse)
        }
