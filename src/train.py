"""
Training pipeline for robotics LSTM with NaN prevention
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import time
from tqdm import tqdm
import json
import os


class Trainer:
    """
    Training orchestrator for robotics LSTM model with robust NaN handling
    """

    def __init__(self, device='cpu', batch_size=16, learning_rate=1e-3,
                 weight_decay=1e-5, epochs=20, sequence_length=10,
                 hidden_dim=128, num_layers=3, max_grad_norm=1.0,
                 save_dir='./models', log_dir='./logs'):
        """
        Args:
            device: 'cuda' or 'cpu'
            batch_size: Training batch size
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            epochs: Number of training epochs
            sequence_length: Length of action/obs sequences
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            max_grad_norm: Gradient clipping threshold
            save_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_grad_norm = max_grad_norm

        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)

        # Model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None

        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.start_time = None
        self.nan_count = 0

    def create_model(self, action_dim, obs_dim, loss_fn):
        """Create model with given dimensions"""
        from model import RoboticsLSTM, RoboticsLoss

        self.model = RoboticsLSTM(
            action_dim=action_dim,
            obs_dim=obs_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.3
        ).to(self.device)

        self.loss_fn = loss_fn if loss_fn is not None else RoboticsLoss()

        # Optimizer with weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999)
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        # Count parameters
        params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("\\n🧠 Model Statistics:")
        print(f"   Parameters: {params:,}")
        print(f"   Device: {self.device}")
        print(f"   Hidden dim: {self.hidden_dim}")
        print(f"   Num layers: {self.num_layers}")

    def train_epoch(self, train_loader):
        """Train for one epoch with robust NaN handling"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move to device
                action_seq = batch['action_seq'].to(self.device)
                obs_seq = batch['obs_seq'].to(self.device)
                next_action = batch['next_action'].to(self.device)

                # Validate inputs
                if (torch.isnan(action_seq).any() or
                        torch.isnan(obs_seq).any()):
                    msg = f"⚠️  NaN detected in batch {batch_idx}"
                    print(msg + ", skipping...")
                    continue

                # Forward pass
                self.optimizer.zero_grad()
                pred_action, _ = self.model(action_seq, obs_seq)

                # Validate prediction
                if torch.isnan(pred_action).any():
                    print(f"⚠️  NaN in model output at batch {batch_idx}")
                    self.nan_count += 1
                    continue

                # Compute loss
                loss = self.loss_fn(pred_action, next_action)

                # Check loss validity
                if torch.isnan(loss) or torch.isinf(loss):
                    msg = f"⚠️  NaN/Inf loss at batch {batch_idx}: "
                    print(msg + f"{loss.item()}")
                    self.nan_count += 1
                    continue

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()

                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            except Exception as e:
                print(f"⚠️  Error at batch {batch_idx}: {str(e)}")
                continue

        avg_loss = total_loss / max(num_batches, 1)
        if num_batches == 0:
            avg_loss = float('inf')

        return avg_loss

    def validate(self, val_loader):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            for batch in pbar:
                try:
                    action_seq = batch['action_seq'].to(self.device)
                    obs_seq = batch['obs_seq'].to(self.device)
                    next_action = batch['next_action'].to(self.device)

                    # Forward pass
                    pred_action, _ = self.model(action_seq, obs_seq)

                    # Compute loss
                    loss = self.loss_fn(pred_action, next_action)

                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_loss += loss.item()
                        num_batches += 1

                    pbar.set_postfix({'loss': f'{loss.item():.6f}'})

                except Exception as e:
                    print(f"⚠️  Validation error: {str(e)}")
                    continue

        avg_loss = total_loss / max(num_batches, 1)
        if num_batches == 0:
            avg_loss = float('inf')

        return avg_loss

    def fit(self, train_loader, val_loader, action_dim, obs_dim, loss_fn=None):
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            action_dim: Action dimension
            obs_dim: Observation dimension
            loss_fn: Loss function (optional)
        """
        print("\n" + "="*60)
        print("🚀 STARTING ROBOTICS LSTM TRAINING")
        print("="*60)
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")

        self.start_time = time.time()

        # Create model
        self.create_model(action_dim, obs_dim, loss_fn)

        # Training loop
        for epoch in range(self.epochs):
            print(f"\n📍 Epoch {epoch + 1}/{self.epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Scheduler step
            self.scheduler.step()

            # Print metrics
            print(f"   Train Loss: {train_loss:.6f}")
            print(f"   Val Loss: {val_loss:.6f}")
            print(f"   LR: {self.scheduler.get_last_lr()[0]:.2e}")

            if self.nan_count > 0:
                print(f"   ⚠️  NaN occurrences: {self.nan_count}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                best_path = self.save_dir / 'best.pt'
                self._save_checkpoint(best_path, epoch, val_loss)
                print(f"   ✓ Best model saved: {best_path}")

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                ckpt_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pt'
                self._save_checkpoint(ckpt_path, epoch, val_loss)

        # Training complete
        elapsed = time.time() - self.start_time
        print(f"\n✓ Training complete in {elapsed/60:.2f} minutes")
        print(f"   Best epoch: {self.best_epoch + 1}")
        print(f"   Best val loss: {self.best_val_loss:.6f}")

        # Save logs
        self._save_logs()

    def _save_checkpoint(self, path, epoch, val_loss):
        """Save checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
        }, path)

    def _save_logs(self):
        """Save training logs"""
        logs = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'best_epoch': self.best_epoch + 1,
            'best_val_loss': float(self.best_val_loss),
            'final_train_loss': float(self.train_losses[-1])
            if self.train_losses else None,
            'final_val_loss': float(self.val_losses[-1])
            if self.val_losses else None,
            'training_time_minutes': (time.time() - self.start_time) / 60,
        }

        log_path = self.log_dir / 'training_logs.json'
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2)

        print(f"✓ Logs saved: {log_path}")
