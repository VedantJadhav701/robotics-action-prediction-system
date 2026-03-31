# Use official PyTorch image as base
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configure pip for better reliability
RUN pip config set global.index-url https://pypi.org/simple/ && \
    pip config set global.retries 5 && \
    pip config set global.timeout 120

# Copy requirements
COPY requirements-prod.txt .

# Install Python dependencies with retry
RUN pip install --no-cache-dir --default-timeout=120 -r requirements-prod.txt || \
    pip install --no-cache-dir --default-timeout=120 -r requirements-prod.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
