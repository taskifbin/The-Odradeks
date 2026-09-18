# syntax=docker/dockerfile:1

# ── Stage 1: Builder ──
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system dependencies required for compiling PuLP/CBC solver bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies into a virtual env inside the container
# This isolates them so we can copy just the site-packages later
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install curl for HEALTHCHECK and runtime diagnostics
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY ./app ./app

# Expose the port FastAPI will listen on
EXPOSE 8000

# Set environment variables defaults (overridable by judge/host)
ENV PYTHONUNBUFFERED=1
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000

# Healthcheck to ensure service is ready before traffic hits it
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

