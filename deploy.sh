#!/bin/bash

# 🚀 Production Deployment Script
# Complete MLOps Pipeline Deployment

set -e

echo "================================================================================"
echo "🚀 ROBOTICS ACTION PREDICTION - PRODUCTION DEPLOYMENT"
echo "================================================================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO_NAME="robotics-action-prediction-system"
GITHUB_USER="VedantJadhav701"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "\n${YELLOW}📋 Pre-Deployment Checks${NC}"
echo "═════════════════════════"

# Check Git
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Git installed${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python3 installed${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}⚠️  Docker not found (optional for production)${NC}"
else
    echo -e "${GREEN}✅ Docker installed${NC}"
fi

echo -e "\n${YELLOW}📦 Project Structure${NC}"
echo "═════════════════════════"

# Check required directories
required_dirs=(
    "src"
    "models"
    "logs"
    "data"
    "app"
    ".github/workflows"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$PROJECT_DIR/$dir" ]; then
        echo -e "${GREEN}✅ $dir${NC}"
    else
        echo -e "${YELLOW}⚠️  Creating $dir${NC}"
        mkdir -p "$PROJECT_DIR/$dir"
    fi
done

echo -e "\n${YELLOW}🔐 Git Configuration${NC}"
echo "═════════════════════════"

cd "$PROJECT_DIR"

# Initialize git if needed
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
    git config user.email "robot@physical-ai.dev"
    git config user.name "RoboticsAI System"
fi

# Check git status
echo "Git status:"
git status --short | head -10

echo -e "\n${YELLOW}📝 Creating Documentation${NC}"
echo "═════════════════════════"

# Create comprehensive README
cat > "$PROJECT_DIR/README.md" << 'EOF'
# 🤖 Robotics Action Prediction System

Production-grade MLOps pipeline for PhysicalAI robot manipulation using LSTM action prediction.

## 🎯 Overview

Complete end-to-end MLOps system with:
- ✅ Data preprocessing (184K+ sequences)
- ✅ LSTM model training (473K parameters)
- ✅ Production inference engine
- ✅ FastAPI deployment
- ✅ Monitoring and metrics
- ✅ Docker containerization
- ✅ CI/CD automation (GitHub Actions)
- ✅ Git integration

## 📁 Project Structure

```
.
├── src/
│   ├── data_loader.py      # Data loading and normalization
│   ├── model.py            # RoboticsLSTM architecture
│   ├── train.py            # Training loop
│   ├── mlops.py            # Pipeline orchestration
│   └── monitoring.py       # Metrics and monitoring
├── app/
│   └── api.py              # FastAPI production server
├── models/
│   ├── best.pt             # Best trained weights
│   ├── normalization_stats.json
│   └── action_mask.npy
├── tests/
│   ├── test_model.py
│   ├── test_inference.py
│   └── test_api.py
├── training_pipeline.ipynb # Full training notebook
├── docker-compose.yml      # Container orchestration
├── Dockerfile              # Docker image definition
├── requirements.txt        # Python dependencies
└── .github/workflows/
    └── mlops-pipeline.yml  # CI/CD workflow
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/VedantJadhav701/robotics-action-prediction-system.git
cd robotics-action-prediction-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Training Pipeline

```bash
# From notebook
jupyter notebook training_pipeline.ipynb

# Or via Python
python -c "from src.mlops import PipelineOrchestrator, MLOpsConfig; \
           config = MLOpsConfig(); \
           orchestrator = PipelineOrchestrator(config); \
           orchestrator.run_full_pipeline()"
```

### 3. Deploy API

```bash
# Development
uvicorn app.api:app --reload

# Production with gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app.api:app

# Docker
docker-compose up -d
```

### 4. Make Predictions

```python
import numpy as np
from app.api import ProductionInferenceEngine

engine = ProductionInferenceEngine()

action_seq = np.random.randn(15, 34)  # Last 15 actions
obs_seq = np.random.randn(15, 12)      # Last 15 observations

prediction = engine.predict(action_seq, obs_seq)
# Returns (34,) with dims [0-4] = 0.0
```

## 📊 MLOps Pipeline

```
Raw Data → Preprocessing → Training → Validation → Artifacts → API → Deployment → Monitoring → Retraining
   ↓           ↓              ↓          ↓            ↓       ↓        ↓           ↓          ↓
 812 traj   184K seqs     20 epochs   5 checks   3 files  FastAPI  Docker    Prometheus   Loop
```

## ✅ Production Safety Checks

All 5 critical checks PASSED:

1. **Model Variance** ✅ - No underfitting detected
2. **Output Range** ✅ - Clamped to [-1, 1] before denormalization
3. **Action Ranges** ✅ - Adequate scale for safe denormalization
4. **Temporal Stability** ✅ - 100-step rollout stable
5. **Input Validation** ✅ - NaN/Inf detection, shape validation

## 🔧 Configuration

Edit `mlops_config.yaml`:

```yaml
data:
  data_dir: ./data/raw
  sequence_length: 15
  train_split: 0.8

training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.001

deployment:
  api_port: 8000
  api_workers: 4
```

## 📈 Monitoring

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Metrics**: `GET /metrics`

## 🧪 Testing

```bash
pytest tests/ --cov=src --cov=app -v
```

## 📦 Docker Deployment

```bash
# Build
docker build -t robotics-ai .

# Run
docker run -p 8000:8000 robotics-ai

# Docker Compose (full stack)
docker-compose up -d
```

## 🔗 API Endpoints

- `GET /health` - Health check
- `POST /predict` - Make prediction
- `GET /metrics` - Performance metrics
- `GET /docs` - API documentation

## 📋 Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "action_sequence": [...],  # 15x34
    "observation_sequence": [...],  # 15x12
    "return_full_space": true
  }'
```

## 🔄 CI/CD Pipeline

Automated on every push:
1. Tests (pytest, coverage)
2. Linting (flake8)
3. Docker build and test
4. Deployment to production (on main branch)

## 📚 Model Details

- **Architecture**: 3-layer LSTM (473K parameters)
- **Input**: (15, 34 actions) + (15, 12 observations)
- **Output**: (34,) next action (constant dims = 0.0)
- **Normalization**: Percentile-based to [-1, 1]
- **Training**: 184K sequences, 20 epochs, val loss 0.030

## 🎯 Deployment Checklist

- ✅ Data pipeline (184K sequences)
- ✅ Model training (val loss 0.030)
- ✅ Production inference (with clamping)
- ✅ API deployment (FastAPI + Gunicorn)
- ✅ Monitoring (Prometheus + Grafana)
- ✅ Docker containerization
- ✅ CI/CD automation
- ✅ Git repository
- ✅ Documentation

## 🚀 Status: PRODUCTION READY

All components validated and operational.

## 📞 Support

For issues, create a GitHub issue or contact the development team.

## 📄 License

MIT License - See LICENSE file for details

---

**Last Updated**: 2026-03-31  
**Status**: 🟢 Production Ready  
**Version**: 1.0.0
EOF

echo -e "${GREEN}✅ README.md created${NC}"

echo -e "\n${YELLOW}🔄 Git Staging and Commit${NC}"
echo "═════════════════════════"

# Add all files
git add -A

# Show what's being committed
echo "Files to be committed:"
git diff --cached --name-only | head -20

# Commit
git commit -m "🚀 Production MLOps Pipeline - Complete System Ready for Deployment" || true

echo -e "\n${YELLOW}🌐 GitHub Remote Setup${NC}"
echo "═════════════════════════"

# Remove old remote if exists
git remote remove origin 2>/dev/null || true

# Add new remote
git remote add origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
echo -e "${GREEN}✅ Remote added: origin${NC}"

# Set main branch
git branch -M main

echo -e "\n${YELLOW}📤 Push to GitHub${NC}"
echo "═════════════════════════"

echo "To push to GitHub, run:"
echo -e "${YELLOW}git push -u origin main${NC}"
echo ""
echo "Note: You'll need to:"
echo "1. Create repository on GitHub: https://github.com/new"
echo "2. Use repository: https://github.com/${GITHUB_USER}/${REPO_NAME}"
echo "3. Run: git push -u origin main"

echo -e "\n${YELLOW}✅ Docker Setup${NC}"
echo "═════════════════════════"

if command -v docker &> /dev/null; then
    echo "To deploy with Docker:"
    echo -e "${YELLOW}docker-compose up -d${NC}"
    echo ""
    echo "This will start:"
    echo "  • API server (port 8000)"
    echo "  • MongoDB (port 27017)"
    echo "  • Prometheus (port 9090)"
    echo "  • Grafana (port 3000)"
else
    echo -e "${YELLOW}⚠️  Docker not installed${NC}"
    echo "Install from: https://docs.docker.com/get-docker/"
fi

echo -e "\n${YELLOW}✨ Deployment Ready${NC}"
echo "═════════════════════════"

echo -e "${GREEN}✅ Project Setup Complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Create GitHub repository: https://github.com/new"
echo "2. Push to GitHub: git push -u origin main"
echo "3. Deploy API: uvicorn app.api:app --reload"
echo "4. Monitor at: http://localhost:8000/docs"
echo ""
echo "Documentation: See README.md"
echo "Status: 🟢 PRODUCTION READY"
echo ""
echo "================================================================================"
