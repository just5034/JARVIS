FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/

RUN python3.12 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir ".[serving]"

# --- Runtime stage ---
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY src/ src/
COPY configs/ configs/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "jarvis", "serve", "--config", "/app/configs"]
