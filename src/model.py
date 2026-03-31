"""
Robotics LSTM model with bounded activations and proper initialization
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class RoboticsLSTM(nn.Module):
    """
    LSTM model for robotics with stable training properties
    """

    def __init__(self, action_dim, obs_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection with layer norm (prevents large inputs)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.ln_action = nn.LayerNorm(hidden_dim)
        self.ln_obs = nn.LayerNorm(hidden_dim)

        # LSTM with proper initialization
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Output projection with layer norm
        self.ln_lstm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization for LSTM"""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                init.orthogonal_(param, gain=1.0)
            elif "weight_hh" in name:
                init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.constant_(param, 0.1)

        # Initialize linear layers
        for module in [self.action_proj, self.obs_proj, self.output_proj]:
            init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, action_seq, obs_seq):
        """
        Forward pass with bounded activations

        Args:
            action_seq: (batch, seq_len, action_dim)
            obs_seq: (batch, seq_len, obs_dim)

        Returns:
            pred_action: (batch, action_dim)
            hidden_state: LSTM hidden states
        """
        batch_size, seq_len, _ = action_seq.shape

        # Project inputs
        action_proj = self.ln_action(torch.relu(self.action_proj(action_seq)))
        obs_proj = self.ln_obs(torch.relu(self.obs_proj(obs_seq)))

        # Clamp projections to prevent explosion
        action_proj = torch.clamp(action_proj, -5.0, 5.0)
        obs_proj = torch.clamp(obs_proj, -5.0, 5.0)

        # Concatenate
        lstm_input = torch.cat([action_proj, obs_proj], dim=-1)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)

        # Use last hidden state
        lstm_out = lstm_out[:, -1, :]  # (batch, hidden_dim)

        # Apply layer norm
        lstm_out = self.ln_lstm(lstm_out)

        # Clamp to prevent explosion
        lstm_out = torch.clamp(lstm_out, -5.0, 5.0)

        # Output projection
        pred_action = self.output_proj(lstm_out)  # (batch, action_dim)

        # Clamp output to [-1, 1]
        pred_action = torch.clamp(pred_action, -1.0, 1.0)

        return pred_action, (h_n, c_n)


class RoboticsLoss(nn.Module):
    """
    Stable loss function for robotics
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, pred, target):
        """
        Compute loss with stability checks

        Args:
            pred: (batch, action_dim) - predicted actions
            target: (batch, action_dim) - target actions

        Returns:
            loss: scalar loss
        """
        # Ensure inputs are valid
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clamp to valid range
        pred = torch.clamp(pred, -1.0, 1.0)
        target = torch.clamp(target, -1.0, 1.0)

        # Compute MSE loss
        loss = self.mse_loss(pred, target)

        # Ensure loss is valid
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        return loss


def count_parameters(model):
    """Count total parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
