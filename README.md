# NVIDIA PhysicalAI - Robotics Action Prediction

A deep learning system for predicting robot actions from state and observation sequences using LSTM networks trained on real NVIDIA PhysicalAI dataset.

## 🎯 Project Overview

This project implements an end-to-end pipeline for training and deploying a robotics action prediction model:

- **Model**: RoboticsLSTM (3-layer LSTM with action/observation fusion)
- **Data**: NVIDIA PhysicalAI Robotics Manipulation Objects dataset
- **Task**: Predict next robot action from 15-step action/observation sequences
- **Framework**: PyTorch with real data processing and validation

## 📊 Dataset

- **Source**: [nvidia/PhysicalAI-Robotics-Manipulation-Objects](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects)
- **Format**: Parquet files (812 episodes)
- **Location**: `data/raw/`
- **Sequence**: Action (34-dim) + Observation (13-dim) per timestep
- **Size**: ~180K+ training samples

## 🏗️ Architecture

```
RoboticsLSTM
├── Input: action_seq (batch, 15, 34) + obs_seq (batch, 15, 13)
├── LSTM layers: 128 hidden, 3 stacked layers
├── Fusion: action_feat + obs_feat concatenated
├── FC layers: 256 → 128 → 34-dim output
└── Output: next_action (batch, 34)
```

**Parameters**: ~495K trainable parameters
**Device**: CUDA (RTX 3050 with 4GB VRAM)

## 📁 Project Structure

```
.
├── src/
│   ├── model.py              # RoboticsLSTM architecture
│   ├── data_loader.py        # Parquet loading with data cleaning
│   ├── train.py              # Trainer class with real data support
│   ├── inference.py          # Prediction and deployment
│   └── monitoring.py         # Logging and metrics
├── training_pipeline.ipynb   # Complete training notebook
├── models/
│   ├── best.pt               # Best checkpoint
│   ├── robotics_lstm.onnx    # ONNX export
│   └── model_summary.json    # Architecture metadata
├── data/
│   └── raw/                  # Parquet files (812 episodes)
├── logs/
│   ├── training_logs.json    # Training metrics
│   └── results_comparison.json
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repo-url>
cd nvidia_physical_ai

# Create Python environment
conda create -n robotics python=3.9
conda activate robotics

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Get parquet files from HuggingFace (recommended)
# 1. Visit: https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects
# 2. Download .parquet files (skip .mp4 videos)
# 3. Place in data/raw/
```

### 3. Train Model

```bash
# Start Jupyter notebook
jupyter notebook training_pipeline.ipynb

# Follow the notebook cells:
# 1. Import and setup
# 2. Load real data (data/raw/)
# 3. Create model
# 4. Train (20 epochs)
# 5. Evaluate and inference
```

### 4. Make Predictions

```python
from src.inference import RoboticsInference

inference = RoboticsInference(checkpoint_path='models/best.pt', device='cuda')
action = inference.predict(action_seq, obs_seq)  # (15, 34), (15, 13) → (34,)
```

## 📈 Training

**Model Performance**:
- Best validation loss: ~0.85 MSE
- Training time: ~20 minutes (20 epochs on RTX 3050)
- Convergence: Stable by epoch 5
- Data: 5000+ real trajectories

**Key Features**:
- ✅ Real parquet data loading with validation
- ✅ NaN/Inf cleaning (70% validity threshold)
- ✅ Gradient clipping and conservative learning rate (1e-4)
- ✅ Separate train/validation split
- ✅ Best model checkpointing

## 🔧 Configuration

Edit `training_pipeline.ipynb` to change hyperparameters:

```python
ACTIVE_CONFIG = {
    'hidden_dim': 128,
    'num_layers': 3,
    'sequence_length': 15,
    'learning_rate': 1e-4,
    'batch_size': 16,
}

PHASE = "baseline"  # "sanity" (3 epochs) or "full" (50 epochs)
```

## 📊 Outputs

After training, saved artifacts:
- `models/best.pt` - Best model checkpoint
- `logs/training_logs.json` - Per-epoch metrics
- `logs/results_comparison.json` - Cross-config comparison
- `models/model_summary.json` - Architecture metadata

## 🧪 Testing

```bash
# Test data loader
python test_loader_fix.py

# Run predictions
python test_predictions.py

# Unit tests
pytest tests/
```

## 📚 Key Components

| Module | Purpose |
|--------|---------|
| `src/model.py` | RoboticsLSTM architecture |
| `src/data_loader.py` | Parquet loading + cleaning |
| `src/train.py` | Trainer with real data support |
| `src/inference.py` | Production inference |
| `training_pipeline.ipynb` | Complete end-to-end pipeline |

## 🔍 Troubleshooting

### Data Loading Issues
```python
# Test directly
from src.data_loader import load_dataset_from_parquet
data = load_dataset_from_parquet(Path('data/raw'), sequence_length=15)
```

### GPU/Memory Issues
- Reduce `batch_size` (16 → 8)
- Reduce `hidden_dim` (128 → 64)
- Reduce `num_layers` (3 → 2)

### Training Not Converging
- Check data quality: `python test_loader_fix.py`
- Verify GPU: `nvidia-smi`
- Use conservative LR: `1e-4`
- Increase epochs: `PHASE = "full"`

## 📦 Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy, Pandas
- Matplotlib, Scikit-learn
- HuggingFace Hub

See `requirements.txt` for complete list.

## 📖 API Reference

### Training
```python
from src.train import Trainer

trainer = Trainer(
    device='cuda',
    learning_rate=1e-4,
    epochs=20,
    batch_size=16,
    sequence_length=15,
    hidden_dim=128,
    num_layers=3
)
trainer.train(train_loader, val_loader)
```

### Inference
```python
from src.inference import RoboticsInference

inference = RoboticsInference('models/best.pt', device='cuda')
single = inference.predict(action_seq, obs_seq)      # Single prediction
batch = inference.predict_batch(action_seqs, obs_seqs)  # Batch predictions
```

## 🤝 Contributing

To experiment with new configurations:
1. Update `ACTIVE_CONFIG` in notebook
2. Run training
3. Results automatically saved to `logs/results_comparison.json`

## 📄 License

MIT License

## 🙏 Citation

```bibtex
@dataset{nvidia_physicalai,
  title={NVIDIA PhysicalAI Robotics Dataset},
  author={NVIDIA Research},
  year={2024},
  url={https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects}
}
```

---

**Last Updated**: April 2026
**Status**: Production Ready ✅
