# ─────────────────────────────────────────────────────────────────────────────
# HBLLM Core — Production Multi-Stage Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Build:   docker build -t hbllm-core .
# Run:     docker run -p 8000:8000 hbllm-core
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Rust Builder ─────────────────────────────────────────────
FROM rust:1.80-slim AS rust-builder

WORKDIR /build
COPY Cargo.toml Cargo.lock* ./
COPY rust/ rust/

RUN mkdir -p /build/output/lib && \
    (cargo build --release --manifest-path rust/compute_kernel/Cargo.toml 2>/dev/null && \
     cp rust/compute_kernel/target/release/*.so /build/output/lib/ 2>/dev/null || true)


# ── Stage 2: Python Builder ───────────────────────────────────────────
FROM python:3.11-slim AS python-builder

WORKDIR /build

# System deps for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pip dependencies first (cache-friendly)
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy source and install
COPY . .
RUN pip install --no-cache-dir --prefix=/install .


# ── Stage 3: Production Runtime ───────────────────────────────────────
FROM python:3.11-slim AS runtime

# OCI standard labels
LABEL org.opencontainers.image.title="HBLLM Core" \
      org.opencontainers.image.description="Human Brain-Like Language Model — Cognitive Architecture" \
      org.opencontainers.image.vendor="HBLLM" \
      org.opencontainers.image.source="https://github.com/hbllm/core"

# Runtime-only system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r hbllm && useradd -r -g hbllm -d /app -s /sbin/nologin hbllm

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=python-builder /install /usr/local

# Copy Rust binaries if built (directory always exists, may be empty)
COPY --from=rust-builder /build/output/lib/ /app/lib/

# Copy source code (needed for package imports)
COPY hbllm/ /app/hbllm/
COPY pyproject.toml /app/

# Create data directories with correct ownership
RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R hbllm:hbllm /app

# Environment
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HBLLM_DATA_DIR=/app/data \
    HBLLM_LOG_LEVEL=INFO

# Switch to non-root user
USER hbllm

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

ENTRYPOINT ["python", "-m", "hbllm.serving.launcher"]
CMD ["--server", "all"]
