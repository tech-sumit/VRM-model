# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HOME=/workspace/cache/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates tini jq build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /workspace/vrm
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv pip install --system --no-deps -e . \
    && uv pip install --system --extra distill -r pyproject.toml --resolution=highest

WORKDIR /workspace/vrm
COPY scripts/pod-entrypoint.sh /usr/local/bin/pod-entrypoint
RUN chmod +x /usr/local/bin/pod-entrypoint

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/pod-entrypoint"]
