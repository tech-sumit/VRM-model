# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HOME=/workspace/cache/hf \
    TRANSFORMERS_CACHE=/workspace/cache/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git curl ca-certificates openssh-client tini jq \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /workspace/vrm
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv pip install --system --no-deps -e . \
    && uv pip install --system --extra eval -r pyproject.toml --resolution=highest

# VLMEvalKit (canonical eval harness; not on PyPI under expected name -- pin from git)
ARG VLMEVALKIT_REF=v0.3.5
RUN git clone --depth 1 --branch ${VLMEVALKIT_REF} https://github.com/open-compass/VLMEvalKit.git /opt/VLMEvalKit \
    && uv pip install --system -e /opt/VLMEvalKit

WORKDIR /workspace/vrm
COPY scripts/pod-entrypoint.sh /usr/local/bin/pod-entrypoint
RUN chmod +x /usr/local/bin/pod-entrypoint

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/pod-entrypoint"]
