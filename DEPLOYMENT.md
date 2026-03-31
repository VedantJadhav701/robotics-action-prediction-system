# 🚀 Deployment Guide - PhysicalAI RoboticsLSTM

Complete deployment instructions for production environments.

## 1. Pre-Deployment Checklist

- ✅ Model trained and validated
- ✅ All artifacts in `models/` directory
- ✅ Docker image tested locally
- ✅ Environment variables configured
- ✅ SSL certificates (if needed)
- ✅ Firewall rules updated

## 2. Local Deployment (Development)

### Start Services

```bash
# Terminal 1: Run FastAPI server
python serve.py

# Terminal 2: Test API
curl http://localhost:8000/health
```

**Expected Output:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "api_version": "1.0.0"
}
```

## 3. Docker Deployment (Recommended)

### Build Image

```bash
# Build with tag
docker build -t robotics-lstm:v1.0.0 .

# Or build via docker-compose
docker-compose build
```

### Run Container

**Option A: Direct Docker**
```bash
docker run -d \
  --name robotics-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/logs:/app/logs \
  robotics-lstm:v1.0.0
```

**Option B: Docker-Compose (Recommended)**
```bash
docker-compose up -d
docker logs robotics-lstm-api
```

### Monitor Deployment

```bash
# Check status
docker ps | grep robotics

# View logs
docker logs -f robotics-lstm-api

# Health check
curl http://localhost:8000/health

# Statistics
curl http://localhost:8000/stats
```

## 4. Cloud Deployment

### AWS ECS

```bash
# 1. Push image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag robotics-lstm:v1.0.0 <account-id>.dkr.ecr.us-east-1.amazonaws.com/robotics-lstm:v1.0.0
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/robotics-lstm:v1.0.0

# 2. Create ECS task definition (task-definition.json)
# 3. Create ECS service
# 4. Configure load balancer
```

### Google Cloud Run

```bash
# 1. Push to GCR
gcloud auth configure-docker
docker tag robotics-lstm:v1.0.0 gcr.io/<project>/robotics-lstm:v1.0.0
docker push gcr.io/<project>/robotics-lstm:v1.0.0

# 2. Deploy to Cloud Run
gcloud run deploy robotics-lstm \
  --image gcr.io/<project>/robotics-lstm:v1.0.0 \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --timeout 60 \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
# 1. Push to ACR
az acr build --registry <name> --image robotics-lstm:v1.0.0 .

# 2. Deploy
az container create \
  --resource-group <group> \
  --name robotics-api \
  --image <name>.azurecr.io/robotics-lstm:v1.0.0 \
  --ports 8000 \
  --environment-variables CUDA_VISIBLE_DEVICES=0
```

### Kubernetes

```bash
# 1. Create deployment YAML
kubectl apply -f k8s/deployment.yaml

# 2. Create service
kubectl apply -f k8s/service.yaml

# 3. Monitor
kubectl get pods
kubectl logs -f <pod-name>
```

**Example Kubernetes deployment (k8s/deployment.yaml):**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: robotics-lstm
spec:
  replicas: 3
  selector:
    matchLabels:
      app: robotics-lstm
  template:
    metadata:
      labels:
        app: robotics-lstm
    spec:
      containers:
      - name: api
        image: robotics-lstm:v1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            nvidia.com/gpu: "1"
          limits:
            memory: "4Gi"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
```

## 5. Performance Optimization

### API Optimization

```python
# In serve.py - Add caching
from functools import lru_cache

@lru_cache(maxsize=128)
def get_normalization_stats():
    # Cache loaded stats
    pass

# Batch processing for multiple predictions
# POST /predict-batch endpoint available
```

### Infrastructure Optimization

```bash
# Set worker processes
python -m uvicorn serve:app --workers 4 --worker-class uvicorn.workers.UvicornWorker

# Use gunicorn for better performance
gunicorn -w 4 -k uvicorn.workers.UvicornWorker serve:app
```

### GPU Optimization

```bash
# Enable mixed precision
CUDA_VISIBLE_DEVICES=0 python serve.py

# Monitor GPU
nvidia-smi dmon

# Profile
python -m cProfile serve.py
```

## 6. Monitoring & Logging

### Application Metrics

```bash
# Prometheus scraping
# Edit prometheus.yml to scrape /metrics endpoint

# Grafana Dashboard
# Import dashboard: robotics-lstm-dashboard.json
```

### Log Aggregation

```bash
# ELK Stack (Elasticsearch, Logstash, Kibana)
# Send logs to centralized location

# CloudWatch (AWS)
# Docker logs automatically forwarded
```

### Error Tracking

```bash
# Integration with error tracking service
# Example: Sentry, DataDog, New Relic
```

## 7. Scaling Strategies

### Horizontal Scaling
```bash
# Multiple instances behind load balancer
# docker-compose with multiple services
# Kubernetes replicas
```

### Vertical Scaling
```bash
# Increase CPU/memory per instance
# Adjust batch size for throughput

# Example: Handle larger batches
# POST /predict-batch with 100 samples at once
```

### Caching Strategy
```bash
# Redis for recent predictions
# In-memory cache for frequently accessed models
# CDN for static assets
```

## 8. Security Hardening

### API Security

```python
# Add authentication
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/predict")
async def predict(request: PredictionRequest, credentials: HTTPAuthCredentials = Depends(security)):
    # Verify token
    pass
```

### Network Security

```bash
# Enable HTTPS
# Rate limiting: 1000 requests/min per IP
# CORS: Restrict to known domains only
# Input validation: Already in place via Pydantic
```

### Container Security

```bash
# Run as non-root user
USER app
RUN useradd -m -u 1000 app

# Use secrets management
# Mount secrets via Docker Secrets or environment files
```

## 9. Troubleshooting

### API Won't Start

```bash
# Check logs
docker logs robotics-lstm-api

# Verify model files
ls -la models/

# Check environment
echo $CUDA_VISIBLE_DEVICES

# Test locally
python serve.py
```

### Health Check Failing

```bash
# Manual health check
curl -v http://localhost:8000/health

# Check if GPU available
nvidia-smi

# Verify model loading
python -c "
import torch
from serve import ProductionRoboticsInferenceEngine
engine = ProductionRoboticsInferenceEngine()
print('✅ Engine loaded')
"
```

### Slow Inference

```bash
# Check latency
curl http://localhost:8000/stats

# Profile inference
# Use nvidia-smi to monitor GPU
nvidia-smi dmon

# Check batch size
# Increase batch processing for throughput
```

## 10. Rollback Procedure

```bash
# Keep previous image tagged
docker images | grep robotics-lstm

# Stop current container
docker stop robotics-lstm-api

# Rollback to previous version
docker run -d \
  --name robotics-lstm-api \
  robotics-lstm:v0.9.0

# Verify
curl http://localhost:8000/health
```

---

## Deployment Checklist

- [ ] Model artifacts validated
- [ ] Docker image built and tested
- [ ] Environment variables configured
- [ ] Health checks passing
- [ ] Monitoring/logging configured
- [ ] Security hardening complete
- [ ] Load testing passed
- [ ] Fallback plan ready
- [ ] Team trained
- [ ] Documentation updated

---

## Quick Reference

| Task | Command |
|------|---------|
| Build | `docker build -t robotics-lstm:v1.0.0 .` |
| Run | `docker-compose up -d` |
| Status | `curl http://localhost:8000/health` |
| Logs | `docker logs robotics-lstm-api` |
| Stop | `docker-compose down` |
| Test | `curl -X POST http://localhost:8000/predict -d {...}` |
| Stats | `curl http://localhost:8000/stats` |

---

**Last Updated:** March 2024
**Status:** ✅ Ready for Production
