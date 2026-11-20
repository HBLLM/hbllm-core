# ─────────────────────────────────────────────────────────────────────────────
# HBLLM Core — Production Multi-Stage Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Build:   docker build -t hbllm-core .
# Run:     docker run -p 8000:8000 -e HBLLM_ENV=production -e HBLLM_JWT_SECRET=... hbllm-core
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Rust Wheel Builder ───────────────────────────────────────
FROM python:3.11-slim AS rust-builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && pip install --no-cache-dir maturin

ENV PATH="/root/.cargo/bin:${PATH}"
ENV PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

COPY Cargo.toml Cargo.lock* ./
COPY rust/ rust/

RUN mkdir -p /build/wheels && \
    for crate in rust/hcir_graph rust/compute_kernel rust/simulation_engine rust/structure_matcher rust/semantic_search rust/perception rust/policy_eval rust/knowledge_graph rust/network_utils rust/confidence rust/concept_extract rust/tokenizer; do \
        if [ -f "$crate/Cargo.toml" ]; then \
            echo "Building wheel for $crate..." && \
            (cd "$crate" && maturin build --release --out /build/wheels); \
        fi; \
    done


# ── Stage 2: Python Builder ───────────────────────────────────────────
FROM python:3.11-slim AS python-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY --from=rust-builder /build/wheels /build/wheels
COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install ".[server]" && \
    pip install --no-cache-dir --prefix=/install /build/wheels/*.whl


# ── Stage 3: Production Runtime ───────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="HBLLM Core" \
      org.opencontainers.image.description="Human-Brain-Inspired Event-Sourced Cognitive OS" \
      org.opencontainers.image.vendor="HBLLM" \
      org.opencontainers.image.source="https://github.com/hbllm/core"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r hbllm && useradd -r -g hbllm -d /app -s /sbin/nologin hbllm

WORKDIR /app

COPY --from=python-builder /install /usr/local
COPY hbllm/ /app/hbllm/
COPY pyproject.toml /app/

RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R hbllm:hbllm /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HBLLM_DATA_DIR=/app/data \
    HBLLM_LOG_LEVEL=INFO

USER hbllm

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

ENTRYPOINT ["python", "-m", "uvicorn", "hbllm.serving.api:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
