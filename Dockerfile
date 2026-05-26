# ─────────────────────────────────────────────────
# Guardian Ear — Multi-Stage Production Dockerfile
# ─────────────────────────────────────────────────
# Build:   docker build -t guardian-ear .
# Run API: docker run -p 8000:8000 guardian-ear api
# Run UI:  docker run -p 8501:8501 guardian-ear dashboard
# ─────────────────────────────────────────────────

FROM python:3.10-slim AS base

# System dependencies for librosa (libsndfile)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create necessary directories
RUN mkdir -p logs alerts .tmp

# ─────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ─────────────────────────────────────────────────
# Entrypoint — select mode via CMD
# ─────────────────────────────────────────────────
# Default: run the FastAPI server
EXPOSE 8000 8501

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh || true

# If entrypoint script doesn't exist, default to API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
