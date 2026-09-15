# SentinX Production Multi-Stage Container
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for high-speed package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency specifications
COPY pyproject.toml .

# Install dependencies into virtual environment
RUN uv venv /app/.venv && \
    uv pip install --no-cache -r pyproject.toml

ENV PATH="/app/.venv/bin:$PATH"

# Copy application source code
COPY . .

# Run pipeline initialization and training
RUN python -m src.pipeline.orchestrator && \
    python -m src.ml.train && \
    python -m src.ml.drift_monitor

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0"]
