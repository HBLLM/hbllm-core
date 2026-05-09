# ─────────────────────────────────────────────────────────────────────────────
# HBLLM Core — Standalone Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Build:   docker build -t hbllm-core .
# Run:     docker run hbllm-core
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# System deps for Rust extensions and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust for native extensions
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy packaging files first to leverage Docker cache
COPY pyproject.toml ./
COPY Cargo.toml ./
COPY Cargo.lock* ./

# Install pip and heavy dependencies first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy source code
COPY . .

# Build Rust extensions and install Python package
RUN cargo build --release 2>/dev/null || true
RUN pip install --no-cache-dir -e .

# Environment setup
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    HBLLM_DATA_DIR=/app/data

# Create data directories
RUN mkdir -p /app/data /app/models /app/logs

EXPOSE 8000

CMD ["python", "-m", "hbllm.serving.launcher", "--server", "all"]
