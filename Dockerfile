# Serving image for the sentiment API (CPU).
# Build:  docker build -t sentiment-api .
# Run:    docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" sentiment-api
# Model weights are mounted at runtime, not baked in, so the image stays small and one
# image can serve any trained checkpoint. To bake them in instead, add: COPY models ./models

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/hf \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only PyTorch wheel: far smaller than the default CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[serve]"
COPY api ./api

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
