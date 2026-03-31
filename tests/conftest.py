"""Pytest configuration and shared fixtures"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_action_sequence():
    """Sample action sequence (15 timesteps, 34 action dims)"""
    return np.random.randn(15, 34).astype(np.float32)


@pytest.fixture
def sample_observation_sequence():
    """Sample observation sequence (15 timesteps, 12 obs dims)"""
    return np.random.randn(15, 12).astype(np.float32)


@pytest.fixture
def sample_next_action():
    """Sample next action (34 dims)"""
    return np.random.randn(34).astype(np.float32)


@pytest.fixture
def device():
    """Test device (CPU for CI compatibility)"""
    return "cpu"


@pytest.fixture
def model(device):
    """RoboticsLSTM model instance"""
    from src.model import RoboticsLSTM

    model = RoboticsLSTM(
        action_dim=34,
        obs_dim=12,
        hidden_dim=128,
        num_layers=3,
        dropout=0.2,
        device=device,
    )
    model.to(device)
    model.eval()
    return model
