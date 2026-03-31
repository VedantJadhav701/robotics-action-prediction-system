# Deploying to Render

Complete guide to deploy your PhysicalAI Robotics Action Prediction system on Render.

## 📋 Prerequisites

1. **Render Account** - Create at [render.com](https://render.com)
2. **GitHub Account** - Already have this ✅
3. **GitHub Personal Access Token** - For API access

## 🚀 Step-by-Step Deployment

### Step 1: Create Render Account & Get API Token

1. Go to [render.com](https://render.com) and sign up
2. Navigate to **Account Settings** → **API Tokens**
3. Click "Create API Token"
4. Copy the token (optional - only needed for manual API control)

**✅ Production Models Included**

Your GitHub repository now includes the production model files:
- ✅ `best.pt` - Main production model (5.7 MB)
- ✅ `normalization_stats.json` - Data normalization parameters
- ✅ `action_mask.npy` - Action dimension filtering mask
- ✅ `action_dimension_metadata.json` - Action metadata
- ✅ `robotics_lstm.onnx` - ONNX format export
- ✅ `model_summary.json` - Model configuration

**These are automatically included in:**
- GitHub repository ✅
- Docker build via `COPY models/ ./models/` ✅
- Render deployment via cloned repository ✅

### Step 2: Connect GitHub Repository

1. Log in to Render Dashboard
2. Click **New** → **Web Service**
3. Select **Build and deploy from a Git repository**
4. Click **Connect GitHub account**
5. Authorize Render to access your repositories
6. Select `robotics-action-prediction-system` repository

### Step 3: Configure Service Settings

**Basic Settings:**
- **Name**: `robotics-action-prediction`
- **Region**: `Oregon` (or your preferred region)
- **Plan**: `Standard` ($12/month) or `Pro` ($25/month)

**Build & Deploy:**
- **Environment**: `Docker`
- **Dockerfile**: `./Dockerfile`
- **Dockerfile Path**: Leave default

**Advanced:**
- **Auto-Deploy**: Enable (redeploy on every push to main)
- **Health Check**: `/health`

### Step 4: (Optional) Add GitHub Secrets

**Note:** Render has built-in auto-deploy from GitHub. When you enable "Auto-Deploy" in Render:

1. Render automatically deploys when you push to the `main` branch
2. No GitHub Secrets needed unless you want manual control
3. GitHub Actions workflow is optional and informational only

**If you want manual deployment control**, add these secrets to your GitHub repository:

**Go to**: GitHub Repository → **Settings** → **Secrets and variables** → **Actions**

```
RENDER_SERVICE_ID=<your-render-service-id>
RENDER_API_TOKEN=<your-api-token-from-step-1>
RENDER_SERVICE_URL=https://robotics-action-prediction.onrender.com
```

**Where to find RENDER_SERVICE_ID:**
- Go to Render Dashboard → select your service
- URL will be: `https://dashboard.render.com/services/<SERVICE_ID>`
- Copy the SERVICE_ID

### Step 5: Deploy

**Option A: Automatic Deployment (Recommended)**

```bash
# Push to main branch - automatically triggers deployment
git push origin main
```

**Option B: Manual Deployment from Render Dashboard**

1. Go to your service in Render Dashboard
2. Click **Manual Deploy** → **Deploy latest commit**

## 📊 Monitoring Deployment

### View Logs

```bash
# In Render Dashboard:
# Service → Logs tab

# Or via CLI (if installed):
render service logs robotics-action-prediction
```

### Check Health

```bash
curl https://robotics-action-prediction.onrender.com/health
```

Expected response:
```json
{
  "status": "ok",
  "timestamp": "2026-04-01T12:34:56.789Z"
}
```

### View API Documentation

```
https://robotics-action-prediction.onrender.com/docs
```

## � Model Management

### Current Strategy: Git-Based (Production)

Models are committed directly to GitHub for production deployment:
- **Advantages**: Simple, works with Render, version controlled
- **Disadvantages**: Not ideal for very large models (>50MB)
- **Files**: ~5.7 MB total (acceptable for git)

### Directory Structure

```
models/
├── best.pt                       # Production inference model (5.7 MB)
├── normalization_stats.json      # Input/output normalization parameters
├── action_mask.npy               # Binary mask for action filtering
├── action_dimension_metadata.json # Action space metadata
├── model_summary.json            # Architecture and hyperparameters
├── robotics_lstm.onnx            # ONNX format export
├── checkpoint_epoch_*.pt         # [IGNORED] Training checkpoints
└── .gitkeep                       # Git placeholder
```

### Future Optimization: Alternative Strategies

For models larger than 50MB, consider these alternatives:

**Option 1: Git LFS (Large File Storage)**
```bash
git lfs install
git lfs track "*.pt"
```

**Option 2: DVC (Data Version Control)**
```bash
dvc add models/best.pt
dvc push  # Push to remote storage (S3, etc.)
```

**Option 3: S3 Download on Deploy**
```dockerfile
RUN wget https://s3.bucket.com/models/best.pt -O ./models/best.pt
```

The following are automatically configured:

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTHONUNBUFFERED` | `1` | Real-time logging |
| `PORT` | `8000` | API port |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device (if available) |

**Add custom variables in Render Dashboard:**
- Service → **Environment**
- Add key-value pairs as needed

## 🐳 Docker Configuration

**Dockerfile optimizations for Render:**

✅ Multi-stage builds (reduces image size)
✅ Health checks (automatic restart on failure)
✅ Production dependencies only (`requirements-prod.txt`)
✅ Non-root user (security best practice)

## 🔒 Security Best Practices

1. **Never commit secrets** - Use GitHub Secrets
2. **Use HTTPS** - Render provides free SSL certificates
3. **Enable CORS properly** - Currently allows all origins
4. **Rate limiting** - Consider adding for production
5. **API authentication** - Consider adding API key auth

## 📈 Scaling

Current configuration:
- **Min instances**: 1
- **Max instances**: 3 (auto-scale on demand)
- **Cost**: ~$12/month base + $0.50/hour per additional instance

## 🛑 Troubleshooting

### Service won't start

1. Check logs in Render Dashboard
2. Verify `requirements-prod.txt` has all dependencies
3. Check port (should be 8000)
4. Review Dockerfile CMD

### Health checks failing

```bash
# Test locally
docker build -t robotics-api .
docker run -p 8000:8000 robotics-api
curl http://localhost:8000/health
```

### Deployment timeout

1. Increase timeout in `.github/workflows/deploy-render.yml`
2. Check your Dockerfile for long-running steps
3. Optimize Docker layers (cache better)

### Out of memory

1. Check service resource usage in Render Dashboard
2. Optimize model loading (lazy loading)
3. Upgrade plan or reduce max instances

## 📜 Useful Commands

```bash
# View deployment logs
render logs robotics-action-prediction

# Check service status
render service describe robotics-action-prediction

# Manual deployment
render deploy robotics-action-prediction --force

# View service URL
render service url robotics-action-prediction
```

## 🔄 CI/CD Pipeline

**Current workflow:**

```
Push to GitHub
    ↓
GitHub Actions: Lint + Test (mlops-pipeline.yml)
    ↓
On success → Deploy to Render (deploy-render.yml)
    ↓
Render builds Docker image + deploys
    ↓
Auto-scales based on traffic
```

## 📚 Additional Resources

- [Render Documentation](https://render.com/docs)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [GitHub Actions](https://docs.github.com/en/actions)

## ✅ Deployment Checklist

- [ ] Created Render account and API token
- [ ] Connected GitHub to Render
- [ ] Configured service in Render Dashboard
- [ ] Added GitHub Secrets (RENDER_SERVICE_ID, RENDER_API_TOKEN, RENDER_SERVICE_URL)
- [ ] Pushed main branch to trigger first deployment
- [ ] Verified service is running via `/health` endpoint
- [ ] Accessed API docs at `/docs`
- [ ] Tested model inference endpoint
- [ ] Configured monitoring/alerts in Render
- [ ] Set up backup strategy for model files

## 🎯 Next Steps

1. **Monitor performance** - Check Render dashboard for metrics
2. **Set up alerts** - Configure notifications for deployment failures
3. **Backup models** - Store model files in separate storage (e.g., S3)
4. **Add authentication** - Implement API key validation
5. **Rate limiting** - Add request throttling for public APIs
6. **Custom domain** - Point your domain to Render service

---

**Status**: Ready for production deployment ✅

**Last Updated**: April 2026
