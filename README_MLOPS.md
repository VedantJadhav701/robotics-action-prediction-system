# 🚀 PhysicalAI RoboticsLSTM - Production MLOps Pipeline

Complete end-to-end machine learning pipeline for robot action prediction with production inference server.

## 📋 Table of Contents
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Installation](#installation)
- [Training](#training)
- [Deployment](#deployment)
- [API Usage](#api-usage)
- [Monitoring](#monitoring)
- [Development](#development)

---

## 🎯 Quick Start

### Local Development (CPU)
```bash
# Clone and setup
git clone <repo>
cd nvidia_physical_ai
conda create -n robotics python=3.10
conda activate robotics
pip install -r requirements-prod.txt

# Run training pipeline
jupyter notebook training_pipeline.ipynb

# Start inference server
python serve.py
# API available at http://localhost:8000/docs
```

### Docker Deployment (GPU)
```bash
# Build and run
docker-compose up -d

# Check status
docker logs robotics-lstm-api
curl http://localhost:8000/health
```

---

## 🏗️ Architecture

### MLOps Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA (812 episodes)                   │
│           184,290 sequences, 15-step windows                │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              1. PREPROCESSING & ANALYSIS                    │
│  ✅ Load 812 parquet files                                  │
│  ✅ Extract 184,290 sequences                               │
│  ✅ Percentile normalization (1-99th to [-1,1])            │
│  ✅ Identify constant dimensions (0-4)                      │
│  ✅ Create action_mask.npy (29 variable, 5 constant)       │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│           2. TRAINING (20 epochs, trajectory split)         │
│  ✅ 3-layer LSTM (128 hidden, 473K params)                 │
│  ✅ Layer normalization at each stage                       │
│  ✅ Orthogonal weight initialization                        │
│  ✅ Gradient clipping (max_norm=1.0)                       │
│  ✅ Best val loss: 0.030445 (epoch 11)                     │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│          3. VALIDATION (5-point safety checks)              │
│  ✅ Model variance: NOT underfitting                        │
│  ✅ Output clamping: [-1, 1] before denorm                 │
│  ✅ Temporal stability: 100-step rollout stable             │
│  ✅ Long-horizon: minimal drift                             │
│  ✅ Input validation: shape, NaN, range checks              │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│            4. ARTIFACTS (Model + Metadata)                  │
│  📦 models/best.pt (trained weights)                        │
│  📦 models/normalization_stats.json (bounds)               │
│  📦 models/action_mask.npy (dimension mask)                │
│  📦 models/action_dimension_metadata.json                   │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│          5. INFERENCE API (FastAPI + Uvicorn)             │
│  🌐 POST /predict - Single prediction                       │
│  🌐 POST /predict-batch - Batch predictions                 │
│  🌐 GET /health - Health check                             │
│  🌐 GET /stats - Inference statistics                      │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│         6. DEPLOYMENT (Docker + docker-compose)             │
│  🐳 Containerized inference service                         │
│  🐳 GPU support (CUDA 11.8)                                │
│  🐳 Health checks & restart policy                         │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│       7. MONITORING (Prometheus + request stats)            │
│  📊 Request counts & success rates                          │
│  📊 Latency tracking                                        │
│  📊 Model performance metrics                               │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│          8. RETRAINING (Triggered by metrics)               │
│  🔄 Monitor model drift                                     │
│  🔄 Collect new data                                        │
│  🔄 Retrain & validate                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU inference)
- Docker & Docker Compose (for containerized deployment)
- Git

### Local Setup

```bash
# 1. Clone repository
git clone <repo>
cd nvidia_physical_ai

# 2. Create conda environment
conda create -n robotics python=3.10 -y
conda activate robotics

# 3. Install dependencies
pip install -r requirements-prod.txt

# 4. Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

### Docker Setup

```bash
# Build image
docker build -t robotics-lstm:latest .

# Or use docker-compose
docker-compose build
```

---

## 🏋️ Training

### Run Full Pipeline

```bash
jupyter notebook training_pipeline.ipynb
```

**Pipeline Steps:**
1. **Load Data** (Cell 4) - Load 812 parquet files
2. **Verify Normalization** (Cell 5) - Check data preprocessing
3. **Test Forward Pass** (Cell 6) - Verify model architecture
4. **Train Model** (Cell 7) - 20 epochs with validation
5. **Analyze Dimensions** (Cell 13) - Identify constant dims
6. **Production Engine** (Cell 14) - Initialize inference engine
7. **Safety Validation** (Cell 15) - 5-point production checks

### Output Artifacts

```
models/
├── best.pt                              # Best checkpoint (epoch 11)
├── checkpoint_epoch_*.pt                # All checkpoints
├── normalization_stats.json             # Min/max bounds
├── action_mask.npy                      # Dimension mask
└── action_dimension_metadata.json       # Dimension info
```

---

## 🚀 Deployment

### Option 1: Local Development

```bash
# Terminal 1: Start inference server
python serve.py
# Server runs on http://localhost:8000

# Terminal 2: Test with curl
curl -X GET http://localhost:8000/health
```

### Option 2: Docker (Recommended)

```bash
# Start all services
docker-compose up -d

# Check logs
docker logs robotics-lstm-api

# Stop services
docker-compose down
```

### Option 3: Cloud Deployment (AWS/GCP/Azure)

```bash
# Push to container registry
docker tag robotics-lstm:latest <registry>/robotics-lstm:latest
docker push <registry>/robotics-lstm:latest

# Deploy using Kubernetes/ECS/Cloud Run
# See DEPLOYMENT.md for detailed instructions
```

---

## 🌐 API Usage

### Health Check

```bash
curl -X GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "api_version": "1.0.0",
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "action_sequence": {
      "actions": [[...15 actions...]]  # (15, 34)
    },
    "observation_sequence": {
      "observations": [[...15 observations...]]  # (15, 12)
    },
    "return_full_space": true
  }'
```

**Response:**
```json
{
  "next_action": [...34 values...],
  "prediction_time_ms": 25.4,
  "model_version": "1.0.0-prod",
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "action_sequences": [[...], [...], ...],     # (batch, 15, 34)
    "observation_sequences": [[...], [...], ...] # (batch, 15, 12)
  }'
```

### Get Statistics

```bash
curl -X GET http://localhost:8000/stats
```

**Response:**
```json
{
  "total_requests": 1250,
  "successful": 1248,
  "failed": 2,
  "avg_latency_ms": 28.3,
  "success_rate": 0.998
}
```

### API Documentation

Auto-generated interactive docs available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 📊 Monitoring

### Request Metrics

Metrics automatically tracked:
- Total requests
- Successful predictions
- Failed requests
- Average latency
- Success rate

**Access metrics:**
```bash
curl http://localhost:8000/stats
```

### Prometheus Integration

```bash
# Prometheus available at http://localhost:9090
# Query examples:
#   - API uptime
#   - Request latency
#   - Error rates
```

### Log Monitoring

```bash
# Docker logs
docker logs -f robotics-lstm-api

# File logs (if configured)
tail -f logs/inference.log
```

---

## 🔧 Development

### Project Structure

```
nvidia_physical_ai/
├── training_pipeline.ipynb          # Full training + validation
├── serve.py                         # FastAPI inference server
├── requirements-prod.txt            # Production dependencies
├── Dockerfile                       # Container definition
├── docker-compose.yml               # Local deployment
├── .gitignore                       # Git ignore rules
├── README.md                        # This file
│
├── src/
│   ├── model.py                     # RoboticsLSTM architecture
│   ├── data_loader.py               # Data loading & normalization
│   ├── train.py                     # Training loop
│   └── __init__.py
│
├── models/
│   ├── best.pt                      # Best trained checkpoint
│   ├── normalization_stats.json     # Normalization bounds
│   ├── action_mask.npy              # Dimension mask
│   └── action_dimension_metadata.json
│
├── data/
│   └── raw/                         # Raw parquet files (812 episodes)
│
├── logs/
│   ├── training_logs.json           # Training metrics
│   └── inference.log                # API logs
│
└── PhysicalAI-Robotics-Manipulation-Objects/
    └── [Dataset structure]
```

### Adding New Features

1. **New inference features:** Edit `serve.py`, add endpoints
2. **Model improvements:** Edit `src/model.py`
3. **Data processing:** Edit `src/data_loader.py`
4. **Training changes:** Edit `training_pipeline.ipynb`

### Running Tests

```bash
# Unit tests (when added)
pytest tests/

# Integration tests
python test_api_integration.py

# Load tests
locust -f locustfile.py
```

---

## 🔐 Production Checklist

Before production deployment:

- ✅ Model validation complete (5-point safety checks)
- ✅ API endpoints tested with diverse inputs
- ✅ Docker image built and tested locally
- ✅ Health checks configured
- ✅ Monitoring/logging enabled
- ✅ Error handling in place
- ✅ Input validation strict
- ✅ Rate limiting configured
- ✅ CORS policies set appropriately
- ✅ Git repository initialized & committed

---

## 📈 Performance Metrics

**Model Performance:**
- Validation loss: 0.030445
- Training loss (final): 0.029612
- Best epoch: 11 / 20
- Effective action dimensions: 29 / 34

**Inference Performance:**
- Latency: ~25-30ms (GPU)
- Throughput: ~33-40 predictions/sec
- Memory: ~500MB (model + buffers)

**Stability:**
- 100-step rollout: Stable
- Temporal drift: Minimal (<5%)
- No NaN/Inf issues
- Output clamping: Active

---

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/xyz`
2. Make changes and test locally
3. Commit with clear messages: `git commit -m "Add xyz feature"`
4. Push and create pull request

---

## 📝 License

[Your License Here]

---

## 📞 Support

For issues or questions:
1. Check [DEPLOYMENT.md](DEPLOYMENT.md) for detailed setup
2. Review [PRODUCTION_VALIDATION_REPORT.md](PRODUCTION_VALIDATION_REPORT.md)
3. Check logs: `docker logs robotics-lstm-api`
4. Create issue on GitHub

---

**Last Updated:** March 2024
**Pipeline Status:** ✅ Production Ready
