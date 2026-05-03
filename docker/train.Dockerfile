# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/workspace/cache/hf \
    TRANSFORMERS_CACHE=/workspace/cache/hf \
    WANDB_DIR=/workspace/wandb \
    TORCH_CUDA_ARCH_LIST="8.0;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        build-essential git curl ca-certificates openssh-client \
        ninja-build libaio-dev pkg-config jq tini \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# --- Layer: Python deps from pyproject.toml ---
WORKDIR /workspace/vrm
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv pip install --system --no-deps -e . \
    && uv pip install --system --extra train -r pyproject.toml --resolution=highest

# --- Layer: LLaMA-Factory (pinned commit; declarative SFT) ---
ARG LLAMAFACTORY_REF=v0.9.0
RUN git clone --depth 1 --branch ${LLAMAFACTORY_REF} https://github.com/hiyouga/LLaMA-Factory.git /opt/LLaMA-Factory \
    && uv pip install --system --no-deps -e /opt/LLaMA-Factory \
    && uv pip install --system "transformers>=4.45" "datasets>=2.20" "peft>=0.12" "trl>=0.11" "deepspeed>=0.15"

# --- Layer: flash-attn (slow build, isolated) ---
RUN MAX_JOBS=4 uv pip install --system --no-build-isolation flash-attn==2.6.3

# --- Layer: vLLM (rollout server for GRPO) ---
RUN uv pip install --system "vllm>=0.6.2"

# --- Default workdir + entrypoint ---
WORKDIR /workspace/vrm
COPY scripts/pod-entrypoint.sh /usr/local/bin/pod-entrypoint
RUN chmod +x /usr/local/bin/pod-entrypoint

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/pod-entrypoint"]
