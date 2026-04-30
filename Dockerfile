FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DTS_ENV=production \
    DTS_HOST=0.0.0.0 \
    DTS_WORKERS=2

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (layer-cached)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Source
COPY . .

# Persistent dirs (mount a Railway volume on /app/outputs to keep reports)
RUN mkdir -p outputs uploads cache

# Railway injects $PORT; expose 8000 only as documentation hint
EXPOSE 8000

# Healthcheck — used by Docker; Railway has its own health probe via railway.json
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/api/health" || exit 1

CMD ["python", "run_api.py"]
