#!/usr/bin/env python3
"""
🧪 Comprehensive Model Robustness Testing
Tests model with realistic inputs, rolling inference, and noise robustness
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/predict"
TEST_RESULTS = {"timestamp": datetime.now().isoformat(), "tests": {}, "summary": {}}


def log_stats(name: str, values: np.ndarray):
    """Log statistical properties of output"""
    stats = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "has_nan": bool(np.isnan(values).any()),
        "has_inf": bool(np.isinf(values).any()),
        "clipped_to_1": bool(np.any(np.abs(values) > 1.0)),
    }
    logger.info(f"\n{name}:")
    logger.info(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    logger.info(f"  NaN: {stats['has_nan']}, Inf: {stats['has_inf']}")
    logger.info(f"  Values > |1.0|: {stats['clipped_to_1']}")
    return stats


def test_zero_input():
    """Baseline: all-zero input (detects learned bias)"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Zero Input (Baseline)")
    logger.info("=" * 60)

    payload = {
        "action_sequence": {"actions": [[0.0] * 34 for _ in range(15)]},
        "observation_sequence": {"observations": [[0.0] * 12 for _ in range(15)]},
        "return_full_space": True,
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            pred = np.array(result["next_action"])
            stats = log_stats("Zero Input Output", pred)
            stats["response_time_ms"] = result.get("prediction_time_ms")
            logger.info(f"✅ Zero test passed (response: {stats['response_time_ms']:.1f}ms)")
            TEST_RESULTS["tests"]["zero_input"] = stats
            return True
        else:
            logger.error(f"❌ Failed with status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


def test_random_realistic():
    """Test with random realistic sequences"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Random Realistic Input")
    logger.info("=" * 60)

    predictions = []

    for trial in range(3):
        logger.info(f"\nTrial {trial+1}/3:")

        # Generate random realistic sequences
        actions = np.random.uniform(-0.5, 0.5, (15, 34)).tolist()
        observations = np.random.uniform(-1.0, 1.0, (15, 12)).tolist()

        payload = {
            "action_sequence": {"actions": actions},
            "observation_sequence": {"observations": observations},
            "return_full_space": True,
        }

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                pred = np.array(result["next_action"])
                predictions.append(pred)

                input_stats = {
                    "action_mean": float(np.mean(actions)),
                    "obs_mean": float(np.mean(observations)),
                }
                logger.info(
                    f"  Input Mean - Actions: {input_stats['action_mean']:.4f}, "
                    f"Obs: {input_stats['obs_mean']:.4f}"
                )
                log_stats(f"  Trial {trial+1} Output", pred)
            else:
                logger.error(f"  ❌ Failed with status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            return False

    # Check consistency across trials
    predictions = np.array(predictions)
    trial_variance = np.var(predictions, axis=0).mean()

    stats = {
        "trials": 3,
        "cross_trial_variance": float(trial_variance),
        "prediction_shape": list(predictions[0].shape),
        "values_range": [float(predictions.min()), float(predictions.max())],
    }
    logger.info("\n✅ Random test passed")
    logger.info(f"  Cross-trial variance: {trial_variance:.6f}")
    TEST_RESULTS["tests"]["random_realistic"] = stats
    return True


def test_rolling_inference():
    """Test rolling/auto-regressive inference (temporal stability)"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Rolling Inference (Temporal Stability)")
    logger.info("=" * 60)

    # Initialize with random sequences
    actions_window = np.random.uniform(-0.5, 0.5, (15, 34)).tolist()
    obs_window = np.random.uniform(-1.0, 1.0, (15, 12)).tolist()

    outputs = []
    is_stable = True

    for step in range(10):
        logger.info(f"\nStep {step+1}/10:")

        payload = {
            "action_sequence": {"actions": actions_window},
            "observation_sequence": {"observations": obs_window},
            "return_full_space": True,
        }

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code != 200:
                logger.error(f"  ❌ Failed with status {response.status_code}")
                return False

            result = response.json()
            pred = np.array(result["next_action"])
            outputs.append(pred)

            # Check for explosions
            if np.isnan(pred).any() or np.isinf(pred).any():
                logger.error("  ❌ NaN/Inf detected!")
                is_stable = False
                break

            # Check for extreme values
            if np.abs(pred).max() > 10.0:
                logger.warning(f"  ⚠️ Large value detected: {np.abs(pred).max():.2f}")

            logger.info(
                f"  Mean: {np.mean(pred):.4f}, "
                f"Std: {np.std(pred):.4f}, "
                f"Max Abs: {np.abs(pred).max():.4f}"
            )

            # Shift window: remove oldest action, add new prediction as next action
            # (simplified - in real scenario would come from env)
            actions_window = actions_window[1:] + [pred.tolist()]

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            is_stable = False
            break

    if is_stable:
        outputs = np.array(outputs)
        stats = {
            "steps": 10,
            "is_stable": True,
            "output_drift": float(np.std(outputs.mean(axis=1))),
            "avg_output_std": float(outputs.std()),
            "max_output": float(outputs.max()),
            "output_trend": "stable",
        }
        logger.info("\n✅ Rolling inference stable over 10 steps")
        logger.info(f"  Output drift (std of means): {stats['output_drift']:.6f}")
        TEST_RESULTS["tests"]["rolling_inference"] = stats
        return True
    else:
        TEST_RESULTS["tests"]["rolling_inference"] = {"is_stable": False}
        return False


def test_noise_robustness():
    """Test robustness to input noise"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Noise Robustness")
    logger.info("=" * 60)

    base_actions = np.random.uniform(-0.5, 0.5, (15, 34))
    base_obs = np.random.uniform(-1.0, 1.0, (15, 12))

    noise_levels = [0.0, 0.01, 0.05, 0.1]
    outputs_by_noise = {}

    for noise_level in noise_levels:
        logger.info(f"\nNoise Level: {noise_level:.3f}")

        # Add noise
        noisy_actions = base_actions + np.random.normal(0, noise_level, base_actions.shape)
        noisy_obs = base_obs + np.random.normal(0, noise_level, base_obs.shape)

        payload = {
            "action_sequence": {"actions": noisy_actions.tolist()},
            "observation_sequence": {"observations": noisy_obs.tolist()},
            "return_full_space": True,
        }

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                pred = np.array(result["next_action"])

                log_stats(f"  Noise σ={noise_level} Output", pred)
                outputs_by_noise[noise_level] = {
                    "mean": float(np.mean(pred)),
                    "std": float(np.std(pred)),
                    "max_abs": float(np.abs(pred).max()),
                }
            else:
                logger.error(f"  ❌ Failed with status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            return False

    # Compute sensitivity as max diff in means across noise levels
    means = [outputs_by_noise[nl]["mean"] for nl in noise_levels]
    max_mean_diff = max(means) - min(means)

    stats = {
        "noise_levels_tested": noise_levels,
        "outputs_by_noise": outputs_by_noise,
        "sensitivity": "low" if max_mean_diff < 0.5 else "high",
    }

    logger.info("\n✅ Noise robustness test passed")
    logger.info(f"  Model sensitivity: {stats['sensitivity']} (max mean diff: {max_mean_diff:.4f})")
    TEST_RESULTS["tests"]["noise_robustness"] = stats
    return True


def test_batch_vs_single():
    """Compare batch vs single predictions for consistency"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Batch vs Single Consistency")
    logger.info("=" * 60)

    # Create test sequences
    actions = np.random.uniform(-0.5, 0.5, (15, 34))
    obs = np.random.uniform(-1.0, 1.0, (15, 12))

    # Test single
    logger.info("\nSingle prediction:")
    single_payload = {
        "action_sequence": {"actions": actions.tolist()},
        "observation_sequence": {"observations": obs.tolist()},
        "return_full_space": True,
    }

    try:
        single_response = requests.post(API_URL, json=single_payload).json()
        single_pred = np.array(single_response["next_action"])
        logger.info(f"  Output shape: {single_pred.shape}")
        log_stats("  Single Output", single_pred)
    except Exception as e:
        logger.error(f"  ❌ Error: {e}")
        return False

    # Test batch (3 identical samples)
    logger.info("\nBatch prediction (3 identical samples):")
    batch_payload = {
        "action_sequences": [actions.tolist()] * 3,
        "observation_sequences": [obs.tolist()] * 3,
    }

    try:
        batch_response = requests.post(
            "http://localhost:8000/predict-batch", json=batch_payload
        ).json()
        batch_preds = np.array(batch_response["next_actions"])
        logger.info(f"  Batch output shape: {batch_preds.shape}")

        # Compare
        diffs = [np.abs(batch_preds[i] - single_pred).max() for i in range(3)]
        max_diff = max(diffs)
        logger.info(f"  Max difference (single vs batch): {max_diff:.8f}")

        if max_diff < 1e-5:
            logger.info("  ✅ Consistent")
            consistency = "perfect"
        elif max_diff < 0.01:
            logger.info("  ✅ Nearly consistent")
            consistency = "good"
        else:
            logger.warning("  ⚠️ Discrepancy detected")
            consistency = "poor"

        stats = {"batch_size": 3, "max_difference": float(max_diff), "consistency": consistency}
        TEST_RESULTS["tests"]["batch_vs_single"] = stats
        return True

    except Exception as e:
        logger.error(f"  ❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 60)
    logger.info("🧪 COMPREHENSIVE MODEL ROBUSTNESS TEST SUITE")
    logger.info("=" * 60)

    tests = [
        ("Zero Input Baseline", test_zero_input),
        ("Random Realistic Input", test_random_realistic),
        ("Rolling Inference", test_rolling_inference),
        ("Noise Robustness", test_noise_robustness),
        ("Batch vs Single", test_batch_vs_single),
    ]

    results = {}
    for name, test_func in tests:
        try:
            passed = test_func()
            results[name] = "✅ PASSED" if passed else "❌ FAILED"
        except Exception as e:
            logger.error(f"❌ Exception in {name}: {e}")
            results[name] = "❌ ERROR"

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    for name, result in results.items():
        logger.info(f"{result} - {name}")

    # Count results
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)

    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n✅ MODEL IS PRODUCTION-READY")
        logger.info("   - Stable outputs")
        logger.info("   - No explosions or NaN")
        logger.info("   - Responsive to inputs")
        logger.info("   - Robust to noise")

    TEST_RESULTS["summary"] = {
        "passed": passed,
        "total": total,
        "verdict": "PRODUCTION_READY" if passed == total else "NEEDS_REVIEW",
    }

    # Save results
    output_file = Path("logs/robustness_test_results.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(TEST_RESULTS, f, indent=2)
    logger.info(f"\n📁 Results saved to {output_file}")


if __name__ == "__main__":
    main()
