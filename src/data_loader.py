"""
Data loader for robotics LSTM with robust normalization and validation
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class RoboticsSequenceDataset(Dataset):
    """
    Dataset for robotics sequences with robust normalization
    """

    def __init__(
        self,
        action_sequences,
        obs_sequences,
        next_actions,
        action_dim,
        obs_dim,
        normalize=True,
        action_bounds=None,
        obs_bounds=None,
    ):
        # action_sequences: (num_seq, seq_len, action_dim)
        # obs_sequences: (num_seq, seq_len, obs_dim)
        # next_actions: (num_seq, action_dim)

        self.action_sequences = action_sequences
        self.obs_sequences = obs_sequences
        self.next_actions = next_actions
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.normalize = normalize

        # Store normalization bounds
        self.action_bounds = action_bounds
        self.obs_bounds = obs_bounds

        if normalize:
            self._apply_normalization()

    def _apply_normalization(self):
        """Apply robust normalization to prevent NaN"""
        # Flatten sequences for computing statistics
        actions_flat = self.action_sequences.reshape(-1, self.action_dim)
        obs_flat = self.obs_sequences.reshape(-1, self.obs_dim)
        next_actions_flat = self.next_actions

        # Combine for global statistics
        all_actions = np.vstack([actions_flat, next_actions_flat])

        # Compute normalization bounds
        if self.action_bounds is None:
            action_min = np.percentile(all_actions, 1, axis=0)
            action_max = np.percentile(all_actions, 99, axis=0)
            self.action_bounds = (action_min, action_max)

        if self.obs_bounds is None:
            obs_min = np.percentile(obs_flat, 1, axis=0)
            obs_max = np.percentile(obs_flat, 99, axis=0)
            self.obs_bounds = (obs_min, obs_max)

        # Normalize action sequences and next actions
        action_min, action_max = self.action_bounds
        action_range = action_max - action_min
        action_range = np.where(action_range < 1e-6, 1.0, action_range)

        self.action_sequences = 2.0 * (self.action_sequences - action_min) / action_range - 1.0
        self.action_sequences = np.clip(self.action_sequences, -1.0, 1.0)
        self.next_actions = 2.0 * (self.next_actions - action_min) / action_range - 1.0
        self.next_actions = np.clip(self.next_actions, -1.0, 1.0)

        # Normalize observation sequences
        obs_min, obs_max = self.obs_bounds
        obs_range = obs_max - obs_min
        obs_range = np.where(obs_range < 1e-6, 1.0, obs_range)

        self.obs_sequences = 2.0 * (self.obs_sequences - obs_min) / obs_range - 1.0
        self.obs_sequences = np.clip(self.obs_sequences, -1.0, 1.0)

        # Remove any remaining NaN/Inf
        self.action_sequences = np.nan_to_num(
            self.action_sequences, nan=0.0, posinf=1.0, neginf=-1.0
        )
        self.obs_sequences = np.nan_to_num(self.obs_sequences, nan=0.0, posinf=1.0, neginf=-1.0)
        self.next_actions = np.nan_to_num(self.next_actions, nan=0.0, posinf=1.0, neginf=-1.0)

    def __len__(self):
        return len(self.action_sequences)

    def __getitem__(self, idx):
        return {
            "action_seq": torch.from_numpy(self.action_sequences[idx].astype(np.float32)),
            "obs_seq": torch.from_numpy(self.obs_sequences[idx].astype(np.float32)),
            "next_action": torch.from_numpy(self.next_actions[idx].astype(np.float32)),
        }


def load_parquet_files(
    parquet_dir, max_samples=None, sequence_length=10, return_trajectory_ids=False
):
    """
    Load robotics trajectories from parquet files with validation

    Args:
        parquet_dir: Path to directory containing parquet files
        max_samples: Maximum number of trajectories to load
        sequence_length: Length of sequences to generate
        return_trajectory_ids: If True, also return trajectory IDs for each sequence

    Returns:
        tuple: (sequences, actions, observations, action_dim, obs_dim)
        or (sequences, actions, observations, action_dim, obs_dim, trajectory_ids) if return_trajectory_ids=True
    """
    parquet_dir = Path(parquet_dir)

    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

    parquet_files = sorted(parquet_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    print(f"📂 Found {len(parquet_files)} parquet file(s)")

    if max_samples:
        parquet_files = parquet_files[:max_samples]

    sequences_list = []
    actions_list = []
    observations_list = []
    # trajectory_ids_list = []  # Track which trajectory each sequence comes from

    action_dim = None
    obs_dim = None
    valid_obs_dims = None  # Track which obs dimensions are valid

    for _, parquet_file in enumerate(tqdm(parquet_files, desc="Loading parquet files")):
        try:
            df = pd.read_parquet(parquet_file)

            # Check for required columns
            if "action" not in df.columns or "observation.state" not in df.columns:
                continue

            # Each parquet file is ONE trajectory with multiple timesteps
            # Stack all timesteps from this trajectory
            actions_trajectory = []
            observations_trajectory = []

            for idx in range(len(df)):
                action_val = df.iloc[idx]["action"]
                obs_val = df.iloc[idx]["observation.state"]

                # Convert to numpy array if needed
                if isinstance(action_val, (list, tuple)):
                    action_val = np.array(action_val, dtype=np.float32)
                elif not isinstance(action_val, np.ndarray):
                    action_val = np.array([action_val], dtype=np.float32)
                else:
                    action_val = np.array(action_val, dtype=np.float32)

                if isinstance(obs_val, (list, tuple)):
                    obs_val = np.array(obs_val, dtype=np.float32)
                elif not isinstance(obs_val, np.ndarray):
                    obs_val = np.array([obs_val], dtype=np.float32)
                else:
                    obs_val = np.array(obs_val, dtype=np.float32)

                # Ensure 1D arrays
                action_val = action_val.flatten()
                obs_val = obs_val.flatten()

                actions_trajectory.append(action_val)
                observations_trajectory.append(obs_val)

            if not actions_trajectory:
                continue

            # Stack timesteps into trajectory: (num_timesteps, action_dim)
            actions = np.array(actions_trajectory, dtype=np.float32)
            observations = np.array(observations_trajectory, dtype=np.float32)

            # Set action dimensions from first valid trajectory
            if action_dim is None:
                action_dim = actions.shape[1]

            # Validate action dimensions match
            if actions.shape[1] != action_dim:
                continue

            # Identify valid observation dimensions
            # (those with at least some non-NaN values)
            if valid_obs_dims is None:
                obs_has_valid = np.isfinite(observations).any(axis=0)
                valid_obs_dims = np.where(obs_has_valid)[0]
                obs_dim = len(valid_obs_dims)
                if obs_dim < observations.shape[1]:
                    obs_total = observations.shape[1]
                    obs_nan = obs_total - obs_dim
                    msg = (
                        f"\u26a0\ufe0f  Using {obs_dim}/{obs_total} "
                        f"observation dimensions "
                        f"({obs_nan} always-NaN)"
                    )
                    print(msg)

            # Filter observations to valid dimensions
            observations = observations[:, valid_obs_dims]

            # Remove rows with NaN values in actions
            valid_mask = ~(np.isnan(actions).any(axis=1))
            actions = actions[valid_mask]
            observations = observations[valid_mask]

            # Fill remaining NaN values in observations with 0
            observations = np.nan_to_num(observations, nan=0.0)

            # Only include trajectories with enough timesteps for sequences
            if len(actions) < sequence_length + 1:
                continue

            # Create multiple sequences from this single trajectory
            trajectory_len = len(actions)
            num_sequences = trajectory_len - sequence_length

            # Add this trajectory and its sequence count
            actions_list.append(actions)
            observations_list.append(observations)

            # Record how many sequences we can extract from this trajectory
            for _ in range(num_sequences):
                sequences_list.append(sequence_length)

        except Exception as e:
            print(f"⚠️  Error loading {parquet_file.name}: {str(e)}")
            continue

    if not actions_list:
        raise ValueError("No valid sequences found in parquet files")

    # Now extract actual sequences from trajectories
    all_action_sequences = []
    # all_obs_sequences = []
    all_trajectory_ids = []

    for traj_id, (actions, observations) in enumerate(
        zip(actions_list, observations_list, strict=True)
    ):
        trajectory_len = len(actions)

        # Extract all overlapping sequences of length sequence_length
        for start_idx in range(trajectory_len - sequence_length):
            # Get sequence of length sequence_length
            # and the next action as target
            action_seq = actions[start_idx : start_idx + sequence_length]
            obs_seq = observations[start_idx : start_idx + sequence_length]
            next_action = actions[start_idx + sequence_length]

            all_action_sequences.append((action_seq, obs_seq, next_action))
            all_trajectory_ids.append(traj_id)

    if not all_action_sequences:
        raise ValueError("No valid sequences found in parquet files")

    # Separate sequences and targets
    action_sequences = np.array(
        [s[0] for s in all_action_sequences]
    )  # (num_seq, seq_len, action_dim)
    obs_sequences = np.array([s[1] for s in all_action_sequences])  # (num_seq, seq_len, obs_dim)
    next_actions = np.array([s[2] for s in all_action_sequences])  # (num_seq, action_dim)

    # Create a modified sequences array that stores sequence
    # length for each sample
    sequences_array = np.full(len(all_action_sequences), sequence_length, dtype=np.int32)

    msg = (
        f"\u2713 Extracted {len(all_action_sequences)} sequences "
        f"from {len(actions_list)} trajectories"
    )
    print(msg)
    msg2 = (
        f"  Action sequence shape: {action_sequences.shape}, "
        f"Obs sequence shape: {obs_sequences.shape}"
    )
    print(msg2)

    if return_trajectory_ids:
        return (
            sequences_array,
            action_sequences,
            obs_sequences,
            next_actions,
            action_dim,
            obs_dim,
            np.array(all_trajectory_ids),
        )
    else:
        return (
            sequences_array,
            action_sequences,
            obs_sequences,
            next_actions,
            action_dim,
            obs_dim,
        )


def create_dataloaders(
    batch_size=16,
    sequence_length=10,
    num_workers=0,
    max_samples=None,
    data_dir="./data/raw",
    test_split=0.2,
    normalize=True,
    trajectory_split=True,
):
    """
    Create train/val dataloaders with proper normalization

    Args:
        batch_size: Batch size
        sequence_length: Sequence length
        num_workers: Number of workers for DataLoader
        max_samples: Maximum trajectories to load
        data_dir: Path to parquet files directory
        test_split: Validation split ratio
        normalize: Whether to apply normalization
        trajectory_split: If True, split by trajectory (not by sequence)

    Returns:
        tuple: (train_loader, val_loader, dataset)
    """
    # Load data - returns pre-extracted sequences
    if trajectory_split:
        (
            sequences,
            action_seqs,
            obs_seqs,
            next_actions,
            action_dim,
            obs_dim,
            traj_ids,
        ) = load_parquet_files(
            data_dir,
            max_samples=max_samples,
            sequence_length=sequence_length,
            return_trajectory_ids=True,
        )

        # Split by trajectory ID
        unique_traj_ids = np.unique(traj_ids)
        num_traj = len(unique_traj_ids)
        val_traj_count = max(1, int(num_traj * test_split))

        # Shuffle trajectory IDs
        np.random.seed(42)
        shuffled_traj_ids = np.random.permutation(unique_traj_ids)
        val_traj_set = set(shuffled_traj_ids[:val_traj_count])

        # Create train/val split based on trajectory membership
        train_indices = np.where(~np.isin(traj_ids, list(val_traj_set)))[0]
        val_indices = np.where(np.isin(traj_ids, list(val_traj_set)))[0]

        print(
            f"📊 Trajectory-based split: {len(train_indices)} train sequences, {len(val_indices)} val sequences"
        )
        print(
            f"   Train trajectories: {num_traj - val_traj_count}, Val trajectories: {val_traj_count}"
        )

        # Reorder all sequences to have train first, then val
        all_indices = np.concatenate([train_indices, val_indices])
        action_seqs = action_seqs[all_indices]
        obs_seqs = obs_seqs[all_indices]
        next_actions = next_actions[all_indices]

    else:
        sequences, action_seqs, obs_seqs, next_actions, action_dim, obs_dim = load_parquet_files(
            data_dir,
            max_samples=max_samples,
            sequence_length=sequence_length,
            return_trajectory_ids=False,
        )
        train_indices = None
        val_indices = None

    # Create dataset with normalization
    dataset = RoboticsSequenceDataset(
        action_seqs, obs_seqs, next_actions, action_dim, obs_dim, normalize=normalize
    )

    dataset.action_dim = action_dim
    dataset.obs_dim = obs_dim

    # Apply split
    if trajectory_split:
        # Data is now ordered: train first, then val
        train_size = len(train_indices)
        val_size = len(val_indices)
        from torch.utils.data import Subset

        train_dataset = Subset(dataset, range(train_size))
        val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    else:
        # Random split
        num_samples = len(dataset)
        val_size = int(num_samples * test_split)
        train_size = num_samples - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("\n📊 Data split:")
    print(f"   Train: {len(train_dataset)} sequences")
    print(f"   Val: {len(val_dataset)} sequences")
    print(f"   Action dim: {action_dim}, Obs dim: {obs_dim}")

    return train_loader, val_loader, dataset


def save_normalization_stats(dataset, save_path="./models/normalization_stats.json"):
    """
    Save normalization statistics to JSON for inference

    Args:
        dataset: RoboticsSequenceDataset with normalization bounds
        save_path: Path to save JSON file
    """
    stats = {
        "action_dim": dataset.action_dim,
        "obs_dim": dataset.obs_dim,
        "action_bounds": {
            "min": (
                dataset.action_bounds[0].tolist()
                if hasattr(dataset.action_bounds[0], "tolist")
                else list(dataset.action_bounds[0])
            ),
            "max": (
                dataset.action_bounds[1].tolist()
                if hasattr(dataset.action_bounds[1], "tolist")
                else list(dataset.action_bounds[1])
            ),
        },
        "obs_bounds": {
            "min": (
                dataset.obs_bounds[0].tolist()
                if hasattr(dataset.obs_bounds[0], "tolist")
                else list(dataset.obs_bounds[0])
            ),
            "max": (
                dataset.obs_bounds[1].tolist()
                if hasattr(dataset.obs_bounds[1], "tolist")
                else list(dataset.obs_bounds[1])
            ),
        },
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Normalization stats saved to {save_path}")
    return stats


def load_normalization_stats(stats_path="./models/normalization_stats.json"):
    """
    Load normalization statistics from JSON

    Args:
        stats_path: Path to JSON file

    Returns:
        dict: Normalization statistics
    """
    with open(stats_path) as f:
        stats = json.load(f)

    # Convert lists back to numpy arrays
    stats["action_bounds"] = (
        np.array(stats["action_bounds"]["min"]),
        np.array(stats["action_bounds"]["max"]),
    )
    stats["obs_bounds"] = (
        np.array(stats["obs_bounds"]["min"]),
        np.array(stats["obs_bounds"]["max"]),
    )

    print(f"✓ Normalization stats loaded from {stats_path}")
    return stats
