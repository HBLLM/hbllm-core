---
title: "Deployment Guide — Production HBLLM"
description: "Deploy HBLLM Core to production with Docker, systemd, or Kubernetes. Covers distributed bus, monitoring, and scaling."
---

# Deployment Guide

## Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["python", "-m", "hbllm.serving.api"]
```

```bash
docker build -t hbllm-core .
docker run -p 8000:8000 \
  -e HBLLM_PROVIDER=openai \
  -e OPENAI_API_KEY=sk-... \
  hbllm-core
```

## Distributed Deployment with Redis

For multi-server deployments, use `RedisBus`:

```bash
# Server 1: API + Cognitive Core
HBLLM_BUS_TYPE=redis \
HBLLM_REDIS_URL=redis://redis-host:6379 \
HBLLM_REDIS_HMAC_KEY=your-secret-key \
python -m hbllm.serving.api

# Server 2: Background Workers (Learning, Sleep Cycle)
HBLLM_BUS_TYPE=redis \
HBLLM_REDIS_URL=redis://redis-host:6379 \
HBLLM_REDIS_HMAC_KEY=your-secret-key \
python -m hbllm.serving.worker
```

## Monitoring

HBLLM ships with OpenTelemetry instrumentation:

- Request latency histograms
- Node activation counts
- Memory usage per tenant
- Bus throughput metrics

### Prometheus + Grafana

```bash
HBLLM_OTEL_ENDPOINT=http://otel-collector:4317 \
python -m hbllm.serving.api
```

## Health Checks

The API exposes a health endpoint:

```bash
curl http://localhost:8000/health
# {"status": "healthy", "nodes": 25, "bus": "running"}
```

## Security Checklist

- [x] Set `HBLLM_REDIS_HMAC_KEY` for distributed deployments
- [x] Use `weights_only=True` (enforced automatically)
- [x] Enable per-tenant rate limiting
- [x] Pin adapter revisions to specific Git tags
- [x] Run behind a reverse proxy (nginx/Caddy) with TLS
