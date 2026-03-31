"""Unit tests for the RoboticsLSTM model"""

import numpy as np
import pytest
import torch


class TestRoboticsLSTMModel:
    """Test suite for RoboticsLSTM model"""

    def test_model_initialization(self, model, device):
        """Test that model initializes correctly"""
        assert model is not None
        assert model.action_dim == 34
        assert model.obs_dim == 12
        assert model.hidden_dim == 128
        assert model.num_layers == 3
        # Model should be on eval mode
        assert not model.training

    def test_model_forward_pass(
        self, model, sample_action_sequence, sample_observation_sequence, device
    ):
        """Test forward pass through model"""
        action_tensor = torch.from_numpy(sample_action_sequence).to(device)
        obs_tensor = torch.from_numpy(sample_observation_sequence).to(device)

        # Add batch dimension
        action_tensor = action_tensor.unsqueeze(0)  # (1, 15, 34)
        obs_tensor = obs_tensor.unsqueeze(0)  # (1, 15, 12)

        with torch.no_grad():
            output = model(action_tensor, obs_tensor)

        # Output should be (batch_size, action_dim) = (1, 34)
        assert output.shape == (1, 34)

    def test_batch_prediction(self, model, device):
        """Test batch prediction with multiple sequences"""
        batch_size = 4
        seq_len = 15
        action_dim = 34
        obs_dim = 12

        actions = torch.randn(batch_size, seq_len, action_dim).to(device)
        observations = torch.randn(batch_size, seq_len, obs_dim).to(device)

        with torch.no_grad():
            predictions = model(actions, observations)

        assert predictions.shape == (batch_size, action_dim)

    def test_output_shape_consistency(self, model, device):
        """Test that output shape is consistent across different inputs"""
        with torch.no_grad():
            # Test 1: Single sequence
            actions1 = torch.randn(1, 15, 34).to(device)
            obs1 = torch.randn(1, 15, 12).to(device)
            out1 = model(actions1, obs1)
            assert out1.shape == (1, 34)

            # Test 2: Batch of 8
            actions2 = torch.randn(8, 15, 34).to(device)
            obs2 = torch.randn(8, 15, 12).to(device)
            out2 = model(actions2, obs2)
            assert out2.shape == (8, 34)

    def test_model_reproducibility(self, model, device):
        """Test that same input produces same output (deterministic mode)"""
        torch.manual_seed(42)
        actions = torch.randn(2, 15, 34).to(device)
        obs = torch.randn(2, 15, 12).to(device)

        with torch.no_grad():
            output1 = model(actions, obs)

        with torch.no_grad():
            output2 = model(actions, obs)

        # Should be identical
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_model_eval_mode(self, model):
        """Test that model is in eval mode (no dropout/batch norm effects)"""
        assert not model.training
        for module in model.modules():
            if hasattr(module, "training"):
                # Check that dropout and batch norm are not active
                if isinstance(module, torch.nn.Dropout | torch.nn.BatchNorm1d):
                    assert not module.training
