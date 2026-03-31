#!/usr/bin/env python3
"""Generate test predictions to populate Prometheus metrics"""

import requests
import numpy as np
import time
from datetime import datetime

API_URL = "http://localhost:8000/predict"

def make_prediction():
    """Make a single prediction"""
    # Create random action and observation sequences
    action_sequence = np.random.randn(15, 34).tolist()
    observation_sequence = np.random.randn(15, 12).tolist()
    
    # Format as nested objects per Pydantic schema
    payload = {
        "action_sequence": {
            "actions": action_sequence
        },
        "observation_sequence": {
            "observations": observation_sequence
        },
        "return_full_space": True
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            pred = result.get('next_action', [])
            print(f"✓ Prediction successful | Shape: {len(pred)} | "
                  f"Time: {result.get('prediction_time_ms', 0):.2f}ms")
            return True
        else:
            print(f"✗ API Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Request failed: {e}")
        return False

def main():
    """Make multiple test predictions"""
    print("=" * 70)
    print(" 🚀 GENERATING TEST METRICS FOR PROMETHEUS")
    print("=" * 70)
    print(f"API Endpoint: {API_URL}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Make 5 predictions with small delays
    num_predictions = 5
    successes = 0
    
    for i in range(1, num_predictions + 1):
        print(f"[{i}/{num_predictions}] Making prediction...", end=" ")
        if make_prediction():
            successes += 1
        time.sleep(0.5)
    
    print()
    print("=" * 70)
    print(f"✓ Completed: {successes}/{num_predictions} successful predictions")
    print()
    print("📊 Prometheus Metrics Available:")
    print("  • http://localhost:9090")
    print("  • Try querying: robotics_api_requests_total")
    print("  • Or: rate(robotics_model_inference_seconds_sum[1m])")
    print()
    print("📈 Grafana Dashboard:")
    print("  • http://localhost:3000")
    print("  • Username: admin")
    print("  • Password: admin")
    print("=" * 70)

if __name__ == "__main__":
    main()
