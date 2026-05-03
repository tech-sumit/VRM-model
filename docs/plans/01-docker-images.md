# Sub-plan 01 — Docker images for GHCR

> Index: [`../plan.md`](../plan.md) · Depends on: [00 bootstrap](00-bootstrap.md) · Status: ☐

**Goal:** Three Docker images published to GHCR — `vrm-train`, `vrm-eval`, `vrm-dataprep` — built once in CI, pulled by RunPod pods. Pre-baking dependencies saves ~5 min per pod cold-start.

**Architecture:** Multi-stage Dockerfiles. CUDA 12.4 base for train/eval (PyTorch 2.4 wheels match this). Slim CPU base for dataprep. All images install `vrm` package + relevant uv extras. Images are ~15 GB (train), ~12 GB (eval), ~500 MB (dataprep).

**Tech Stack:** Docker · CUDA 12.4 · PyTorch 2.4 · LLaMA-Factory · TRL · vLLM 0.6 · flash-attn 2.6 · VLMEvalKit · uv inside container.

---

### Task 1: Train image (`docker/train.Dockerfile`)

**Files:**
- Create: `projects/VRM-model/docker/train.Dockerfile`
- Create: `projects/VRM-model/.dockerignore`

- [ ] **Step 1: Write `.dockerignore` to keep image lean**

`projects/VRM-model/.dockerignore`:

```dockerignore
.git
.venv
__pycache__
.pytest_cache
.ruff_cache
*.egg-info
build
dist
data
checkpoints
runs
wandb
outputs
.env
.env.*
docs
tests
*.md
!README.md
```

- [ ] **Step 2: Write `docker/train.Dockerfile`**

`projects/VRM-model/docker/train.Dockerfile`:

```dockerfile
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

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        build-essential git curl ca-certificates openssh-client \
        ninja-build libaio-dev pkg-config jq tini \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# uv (faster than pip for the rest)
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

# --- Layer: flash-attn (slow build, isolate) ---
RUN MAX_JOBS=4 uv pip install --system --no-build-isolation flash-attn==2.6.3

# --- Layer: vLLM (rollout server for GRPO) ---
RUN uv pip install --system "vllm>=0.6.2"

# --- Default workdir + entrypoint ---
WORKDIR /workspace/vrm
COPY scripts/pod-entrypoint.sh /usr/local/bin/pod-entrypoint
RUN chmod +x /usr/local/bin/pod-entrypoint

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/pod-entrypoint"]
```

- [ ] **Step 3: Local sanity build (skip if no Docker locally)**

```bash
cd projects/VRM-model
docker build -f docker/train.Dockerfile -t ghcr.io/tech-sumit/vrm-train:dev .
```

Expected: completes in 8-15 min on first build. If you get `flash-attn` build OOM, increase Docker Desktop RAM to ≥12 GB or set `--build-arg MAX_JOBS=2`.

- [ ] **Step 4: Smoke-test the image**

```bash
docker run --rm --gpus all ghcr.io/tech-sumit/vrm-train:dev python -c "
import torch, vllm, trl, transformers
print(torch.__version__, torch.cuda.is_available())
print('vllm', vllm.__version__, 'trl', trl.__version__, 'transformers', transformers.__version__)
"
```

Expected (on a GPU host): `2.4.0 True` plus version lines.

- [ ] **Step 5: Commit**

```bash
git add projects/VRM-model/docker/train.Dockerfile projects/VRM-model/.dockerignore
git commit -m "vrm: add train Dockerfile (CUDA 12.4 + LF + TRL + vLLM + flash-attn)"
```

---

### Task 2: Eval image (`docker/eval.Dockerfile`)

**Files:**
- Create: `projects/VRM-model/docker/eval.Dockerfile`

- [ ] **Step 1: Write `docker/eval.Dockerfile`**

`projects/VRM-model/docker/eval.Dockerfile`:

```dockerfile
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

# VLMEvalKit (canonical eval harness — pinned commit)
ARG VLMEVALKIT_REF=v0.3.5
RUN git clone --depth 1 --branch ${VLMEVALKIT_REF} https://github.com/open-compass/VLMEvalKit.git /opt/VLMEvalKit \
    && uv pip install --system -e /opt/VLMEvalKit

# qwen-vl-utils for chat template + image preprocessing
RUN uv pip install --system "qwen-vl-utils>=0.0.8"

WORKDIR /workspace/vrm
COPY scripts/pod-entrypoint.sh /usr/local/bin/pod-entrypoint
RUN chmod +x /usr/local/bin/pod-entrypoint

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/pod-entrypoint"]
```

- [ ] **Step 2: Local sanity build**

```bash
docker build -f docker/eval.Dockerfile -t ghcr.io/tech-sumit/vrm-eval:dev .
```

- [ ] **Step 3: Smoke-test**

```bash
docker run --rm ghcr.io/tech-sumit/vrm-eval:dev python -c "
import vlmeval, transformers
print('vlmeval', vlmeval.__version__, 'transformers', transformers.__version__)
"
```

- [ ] **Step 4: Commit**

```bash
git add projects/VRM-model/docker/eval.Dockerfile
git commit -m "vrm: add eval Dockerfile (CUDA runtime + VLMEvalKit pinned)"
```

---

### Task 3: Dataprep image (`docker/dataprep.Dockerfile`)

**Files:**
- Create: `projects/VRM-model/docker/dataprep.Dockerfile`

- [ ] **Step 1: Write `docker/dataprep.Dockerfile`**

`projects/VRM-model/docker/dataprep.Dockerfile`:

```dockerfile
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
```

- [ ] **Step 2: Local sanity build**

```bash
docker build -f docker/dataprep.Dockerfile -t ghcr.io/tech-sumit/vrm-dataprep:dev .
```

- [ ] **Step 3: Smoke-test**

```bash
docker run --rm ghcr.io/tech-sumit/vrm-dataprep:dev python -c "
import anthropic, openai, datasets
print('anthropic', anthropic.__version__, 'openai', openai.__version__)
"
```

- [ ] **Step 4: Commit**

```bash
git add projects/VRM-model/docker/dataprep.Dockerfile
git commit -m "vrm: add dataprep Dockerfile (slim CPU image with anthropic + openai SDKs)"
```

---

### Task 4: Pod entrypoint script (consumed by all three images)

**Files:**
- Create: `projects/VRM-model/scripts/pod-entrypoint.sh`

- [ ] **Step 1: Write the entrypoint that dispatches based on `$VRM_TASK`**

`projects/VRM-model/scripts/pod-entrypoint.sh`:

```bash
#!/usr/bin/env bash
# Pod entrypoint. Dispatches to a vrm.* module based on VRM_TASK env var.
# This script is the SINGLE entrypoint baked into all three images. The
# RunPod pod is started with VRM_TASK=<sft|grpo|rejection|eval|dataprep>
# and per-task env vars (DATA_VERSION, RUN_NAME, CHECKPOINT, ...).
set -euo pipefail

log() { echo "[$(date -Iseconds)] $*"; }
trap 'log "FATAL: line $LINENO failed"; vrm-webhook failure "${VRM_TASK:-?}" "trap line $LINENO" || true; exit 1' ERR

: "${VRM_TASK:?VRM_TASK env var is required (sft|grpo|rejection|eval|dataprep)}"
: "${RUN_NAME:?RUN_NAME env var is required}"

# Pull latest source on every cold start so we always run committed code.
if [[ -n "${VRM_GIT_REPO:-}" ]] && [[ -n "${VRM_GIT_REF:-}" ]]; then
    log "Pulling vrm source from $VRM_GIT_REPO@$VRM_GIT_REF"
    cd /workspace
    rm -rf vrm-src
    git clone --depth 1 --branch "$VRM_GIT_REF" "$VRM_GIT_REPO" vrm-src
    cd vrm-src/projects/VRM-model
    pip install --no-deps -e .
fi

# Budget tripwire daemon (background)
python -m vrm.infra.budget --task "$VRM_TASK" --max-usd "${VRM_MAX_USD:?}" &
BUDGET_PID=$!
trap 'kill $BUDGET_PID 2>/dev/null || true' EXIT

log "Pod entrypoint: VRM_TASK=$VRM_TASK RUN_NAME=$RUN_NAME"
python -m vrm.infra.webhook started "$VRM_TASK" "$RUN_NAME"

case "$VRM_TASK" in
    sft|rejection)
        exec python -m vrm.train.stage1_sft \
            --config "${VRM_CONFIG:?}" \
            --data-version "${DATA_VERSION:?}" \
            --run-name "$RUN_NAME"
        ;;
    grpo)
        exec python -m vrm.train.stage2_grpo \
            --config "${VRM_CONFIG:?}" \
            --sft-checkpoint "${SFT_CHECKPOINT:?}" \
            --data-version "${DATA_VERSION:?}" \
            --run-name "$RUN_NAME"
        ;;
    eval)
        exec python -m vrm.eval.run_vlmevalkit \
            --checkpoint "${CHECKPOINT:?}" \
            --suite "${SUITE:?}" \
            --run-name "$RUN_NAME"
        ;;
    dataprep)
        exec python -m vrm.data.distill \
            --recipe "${VRM_CONFIG:?}" \
            --data-version "${DATA_VERSION:?}"
        ;;
    *)
        log "Unknown VRM_TASK=$VRM_TASK"
        exit 2
        ;;
esac
```

- [ ] **Step 2: Make executable**

```bash
chmod +x projects/VRM-model/scripts/pod-entrypoint.sh
```

- [ ] **Step 3: Commit**

```bash
git add projects/VRM-model/scripts/pod-entrypoint.sh
git commit -m "vrm: add pod entrypoint script (dispatches by VRM_TASK env)"
```

---

## Done when

- [ ] All three Dockerfiles build locally without error.
- [ ] All three images pass their smoke `python -c` check.
- [ ] `pod-entrypoint.sh` is committed and executable.
- [ ] Sub-plan 02 (runpod infra) can reference these images.
- [ ] Sub-plan 10 (CI workflow) will publish them to GHCR with proper tags.
