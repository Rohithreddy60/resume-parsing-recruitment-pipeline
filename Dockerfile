# Multi-stage Dockerfile for Resume Parsing & Recruitment Pipeline
# Stage 1: Builder with spaCy model download
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Stage 2: API Runtime
FROM python:3.12-slim AS api

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY api/ ./api/
COPY pipeline/ ./pipeline/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# Stage 3: Worker Runtime
FROM python:3.12-slim AS worker

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY pipeline/ ./pipeline/

RUN useradd --create-home workeruser
USER workeruser

CMD ["python", "-m", "pipeline.worker"]
