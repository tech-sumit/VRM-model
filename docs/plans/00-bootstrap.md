# Sub-plan 00 — Bootstrap

> Index: [`../plan.md`](../plan.md) · Status: ☐

**Goal:** Create the empty-but-importable Python package structure under `projects/VRM-model/` with uv, ruff, pyright, pytest, a Makefile mirror of CI commands, a `.env.example` template, and an empty package skeleton that other sub-plans extend.

**Architecture:** uv-managed Python 3.11 project. Source under `src/vrm/` (src layout). Tests under `tests/`. Configs under `configs/`. Scripts under `scripts/`. All dev workflows reachable through `make`.

**Tech Stack:** Python 3.11 · uv 0.4+ · ruff · pyright · pytest · GNU make.

---

### Task 1: Create the project skeleton with uv

**Files:**
- Create: `projects/VRM-model/pyproject.toml`
- Create: `projects/VRM-model/.python-version`
- Create: `projects/VRM-model/uv.lock` (generated)
- Create: `projects/VRM-model/src/vrm/__init__.py`
- Create: `projects/VRM-model/src/vrm/py.typed`

- [ ] **Step 1: Verify uv is installed**

```bash
cd projects/VRM-model
uv --version
```
Expected: `uv 0.4.x` or newer. If missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

- [ ] **Step 2: Initialize uv project**

```bash
cd projects/VRM-model
uv init --package --name vrm --python 3.11 --no-readme --no-pin-python
```

This produces `pyproject.toml`, `src/vrm/__init__.py`, and `.python-version`.

- [ ] **Step 3: Replace generated pyproject.toml with the full version**

```toml
[project]
name = "vrm"
version = "0.1.0"
description = "VRM-7B: Visual Reasoning Model training & evaluation toolchain"
requires-python = ">=3.11,<3.12"
authors = [{ name = "Sumit Agrawal", email = "mr.sumitagrawal.17@gmail.com" }]
license = { text = "Apache-2.0" }
dependencies = [
    "pydantic>=2.7",
    "pyyaml>=6.0",
    "click>=8.1",
    "rich>=13.7",
    "httpx>=0.27",
    "tenacity>=8.5",
    "datasets>=2.20",
    "huggingface-hub>=0.24",
    "python-dotenv>=1.0",
    "sympy>=1.13",
    "regex>=2024.5",
    "pillow>=10.4",
]

[project.optional-dependencies]
train = [
    "torch==2.4.0",
    "transformers>=4.45",
    "accelerate>=0.34",
    "peft>=0.12",
    "trl>=0.11",
    "deepspeed>=0.15",
    "bitsandbytes>=0.43",
    "wandb>=0.17",
    "vllm>=0.6.2",
    "flash-attn>=2.6.3",
    "qwen-vl-utils>=0.0.8",
]
eval = [
    "torch==2.4.0",
    "transformers>=4.45",
    "vlmeval>=0.1.0",
]
distill = [
    "anthropic>=0.34",
    "openai>=1.40",
]
dev = [
    "ruff>=0.6",
    "pyright>=1.1.380",
    "pytest>=8.3",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "responses>=0.25",
]

[project.scripts]
vrm = "vrm.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vrm"]

[tool.ruff]
line-length = 110
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"

[tool.pyright]
include = ["src", "tests"]
typeCheckingMode = "standard"
pythonVersion = "3.11"
reportMissingImports = "warning"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --strict-config"
markers = [
    "integration: requires network or large fixtures",
    "gpu: requires CUDA",
]
```

- [ ] **Step 4: Add py.typed marker for typed distribution**

```bash
touch src/vrm/py.typed
```

- [ ] **Step 5: Replace `src/vrm/__init__.py` with version export**

```python
"""VRM-7B: Visual Reasoning Model training & evaluation toolchain."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Run uv lock + sync**

```bash
uv sync --extra dev
```

Expected: writes `uv.lock`, creates `.venv/`, installs pydantic + dev deps. Train/eval/distill extras intentionally NOT installed locally — they're CUDA-heavy and only run inside Docker.

- [ ] **Step 7: Verify importability**

```bash
uv run python -c "import vrm; print(vrm.__version__)"
```
Expected: `0.1.0`

- [ ] **Step 8: Commit**

```bash
git add projects/VRM-model/pyproject.toml projects/VRM-model/uv.lock projects/VRM-model/.python-version projects/VRM-model/src/
git commit -m "vrm: bootstrap uv-managed Python package skeleton"
```

---

### Task 2: Add Makefile, .env.example, .gitignore, README

**Files:**
- Create: `projects/VRM-model/Makefile`
- Create: `projects/VRM-model/.env.example`
- Create: `projects/VRM-model/.gitignore`
- Create: `projects/VRM-model/README.md`

- [ ] **Step 1: Write Makefile (mirror of CI commands)**

`projects/VRM-model/Makefile`:

```make
.PHONY: help sync lint fmt typecheck test smoke clean image-train image-eval image-dataprep \
        data train-sft train-grpo train-rejection eval release

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

sync: ## install/update local deps via uv
	uv sync --extra dev

lint: ## ruff check + format check
	uv run ruff check src tests
	uv run ruff format --check src tests

fmt: ## auto-fix ruff lint + format
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck: ## pyright
	uv run pyright

test: ## pytest unit + integration (no GPU/network)
	uv run pytest -m "not gpu and not integration" --cov=src/vrm --cov-report=term-missing

test-integration: ## pytest including integration markers
	uv run pytest -m "not gpu" --cov=src/vrm --cov-report=term-missing

smoke: ## run base Qwen2.5-VL-7B reference inference (requires GPU)
	bash scripts/smoke-base-inference.sh

clean: ## remove caches, artifacts
	rm -rf .pytest_cache .ruff_cache .venv build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

image-train: ## build train image locally
	docker build -f docker/train.Dockerfile -t ghcr.io/tech-sumit/vrm-train:dev .

image-eval: ## build eval image locally
	docker build -f docker/eval.Dockerfile -t ghcr.io/tech-sumit/vrm-eval:dev .

image-dataprep: ## build dataprep image locally
	docker build -f docker/dataprep.Dockerfile -t ghcr.io/tech-sumit/vrm-dataprep:dev .

# --- Operations against RunPod (mirror of GH Actions workflows) ---
DATA_VERSION ?= dev
RUN_NAME ?= local-$(shell date +%Y%m%d-%H%M%S)

data: ## launch dataprep pod, build SFT+RL shards, push to HF
	uv run python -m vrm.infra.runpod launch-dataprep \
	    --recipe configs/data/sft_recipe.yaml --recipe configs/data/rl_recipe.yaml \
	    --data-version $(DATA_VERSION)

train-sft: ## launch Stage 1 SFT pod
	uv run python -m vrm.infra.runpod launch-train \
	    --stage sft --config configs/stage1_sft_full.yaml \
	    --data-version $(DATA_VERSION) --run-name $(RUN_NAME)

train-grpo: ## launch Stage 2 GRPO pod (requires SFT_CHECKPOINT)
	uv run python -m vrm.infra.runpod launch-train \
	    --stage grpo --config configs/stage2_grpo.yaml \
	    --sft-checkpoint $(SFT_CHECKPOINT) \
	    --data-version $(DATA_VERSION) --run-name $(RUN_NAME)

train-rejection: ## launch Stage 3 rejection-sampled SFT (requires GRPO_CHECKPOINT)
	uv run python -m vrm.infra.runpod launch-train \
	    --stage rejection --config configs/stage3_rejection_sft.yaml \
	    --grpo-checkpoint $(GRPO_CHECKPOINT) \
	    --data-version $(DATA_VERSION) --run-name $(RUN_NAME)

SUITE ?= full
eval: ## launch eval pod for a checkpoint (requires CHECKPOINT)
	uv run python -m vrm.infra.runpod launch-eval \
	    --checkpoint $(CHECKPOINT) --suite $(SUITE)

release: ## tag-and-release flow (manual git tag triggers vrm-release.yml)
	@echo "git tag vrm-7b-vX.Y.Z && git push origin vrm-7b-vX.Y.Z"
```

- [ ] **Step 2: Write .env.example**

`projects/VRM-model/.env.example`:

```dotenv
# === RunPod ===
# Get from https://www.runpod.io/console/user/settings (API Keys)
RUNPOD_API_KEY=
# Pod template/image (set to "ghcr.io/tech-sumit/vrm-train:latest" once CI publishes)
VRM_TRAIN_IMAGE=ghcr.io/tech-sumit/vrm-train:latest
VRM_EVAL_IMAGE=ghcr.io/tech-sumit/vrm-eval:latest
VRM_DATAPREP_IMAGE=ghcr.io/tech-sumit/vrm-dataprep:latest
# RunPod Secure Cloud GPU type id (verify in RunPod console)
VRM_GPU_TYPE_TRAIN=NVIDIA H200
VRM_GPU_COUNT_TRAIN=8
VRM_GPU_TYPE_EVAL=NVIDIA H200
VRM_GPU_COUNT_EVAL=1
# Network volume id (create once in RunPod console, paste here)
VRM_NETWORK_VOLUME_ID=
# Region (must match volume region)
VRM_REGION=US-GA-2

# === HuggingFace Hub ===
# https://huggingface.co/settings/tokens (write scope on tech-sumit org)
HF_TOKEN=
HF_ORG=tech-sumit

# === Teacher APIs (only needed for data-build) ===
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
TEACHER_MODEL_CLAUDE=claude-opus-4-7-thinking-high
TEACHER_MODEL_OPENAI=gpt-5.4-medium

# === Tracking ===
WANDB_API_KEY=
WANDB_PROJECT=vrm-7b
WANDB_ENTITY=tech-sumit

# === Notifications (optional) ===
SLACK_WEBHOOK_VRM=

# === Budget tripwires (USD) ===
VRM_MAX_USD_DATAPREP=500
VRM_MAX_USD_SFT=1500
VRM_MAX_USD_GRPO=8000
VRM_MAX_USD_EVAL=200

# === GitHub (for pod → GH webhook) ===
GH_REPO=tech-sumit/react-webmcp
GH_TOKEN_FOR_DISPATCH=
```

- [ ] **Step 3: Write .gitignore**

`projects/VRM-model/.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
.ruff_cache/
.coverage
coverage.xml
htmlcov/

# Build
build/
dist/
*.whl

# Local env
.env
*.local

# Editor / OS
.DS_Store
.idea/
.vscode/
*.swp

# Datasets / models / weights — never commit large blobs
data/
checkpoints/
runs/
wandb/
outputs/
*.pt
*.bin
*.safetensors
*.gguf
*.parquet

# Notebooks
.ipynb_checkpoints/
*.ipynb_checkpoints
```

- [ ] **Step 4: Write README**

`projects/VRM-model/README.md`:

```markdown
# VRM-7B

Open-weights 7-8B Visual Reasoning Model. SFT cold-start + GRPO RL post-training on Qwen2.5-VL-7B-Instruct.

See:
- [`VRM-7B_model_spec.md`](VRM-7B_model_spec.md) — locked model spec
- [`docs/design.md`](docs/design.md) — toolchain design
- [`docs/plan.md`](docs/plan.md) — implementation plan (index)
- [`docs/runbook.md`](docs/runbook.md) — operations

## Quickstart (local dev)

```bash
cd projects/VRM-model
cp .env.example .env  # fill in tokens
make sync             # install deps (no GPU needed for dev)
make test             # unit tests
make lint             # ruff + format check
```

## Quickstart (RunPod operations)

```bash
make data DATA_VERSION=v1
make train-sft DATA_VERSION=v1 RUN_NAME=sft-2026-05-03
make train-grpo SFT_CHECKPOINT=tech-sumit/vrm-7b-sft-2026-05-03 \
                DATA_VERSION=v1 RUN_NAME=grpo-2026-05-04
make eval CHECKPOINT=tech-sumit/vrm-7b-grpo-2026-05-04 SUITE=full
```

CI/CD via GitHub Actions: see `.github/workflows/vrm-*.yml`.
```

- [ ] **Step 5: Commit**

```bash
git add projects/VRM-model/Makefile projects/VRM-model/.env.example projects/VRM-model/.gitignore projects/VRM-model/README.md
git commit -m "vrm: add Makefile, .env.example, .gitignore, README"
```

---

### Task 3: Stub the package layout (empty but importable subpackages)

**Files:**
- Create: `projects/VRM-model/src/vrm/cli.py`
- Create: `projects/VRM-model/src/vrm/data/__init__.py`
- Create: `projects/VRM-model/src/vrm/data/normalize/__init__.py`
- Create: `projects/VRM-model/src/vrm/data/verifiers/__init__.py`
- Create: `projects/VRM-model/src/vrm/train/__init__.py`
- Create: `projects/VRM-model/src/vrm/eval/__init__.py`
- Create: `projects/VRM-model/src/vrm/infra/__init__.py`
- Create: `projects/VRM-model/tests/__init__.py`
- Create: `projects/VRM-model/tests/unit/__init__.py`
- Create: `projects/VRM-model/tests/integration/__init__.py`
- Create: `projects/VRM-model/tests/conftest.py`

- [ ] **Step 1: Create empty subpackage __init__.py files**

```bash
cd projects/VRM-model
mkdir -p src/vrm/data/normalize src/vrm/data/verifiers src/vrm/train src/vrm/eval src/vrm/infra
mkdir -p tests/unit tests/integration
touch src/vrm/data/__init__.py src/vrm/data/normalize/__init__.py src/vrm/data/verifiers/__init__.py
touch src/vrm/train/__init__.py src/vrm/eval/__init__.py src/vrm/infra/__init__.py
touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py
```

- [ ] **Step 2: Write minimal CLI entrypoint**

`projects/VRM-model/src/vrm/cli.py`:

```python
"""VRM CLI entrypoint. Each subcommand defers to a module under vrm.*"""
from __future__ import annotations

import click

from vrm import __version__


@click.group(help="VRM-7B training & evaluation toolchain")
@click.version_option(__version__)
def main() -> None:
    """Top-level CLI group; subcommands attached lazily."""


@main.command(help="Print package version and exit.")
def version() -> None:
    click.echo(__version__)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write conftest.py with shared fixtures**

`projects/VRM-model/tests/conftest.py`:

```python
"""Shared pytest fixtures for the VRM test suite."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def fixtures_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures"
```

- [ ] **Step 4: Write a smoke test for CLI**

`projects/VRM-model/tests/unit/test_cli.py`:

```python
from click.testing import CliRunner

from vrm.cli import main


def test_version_command_outputs_semver():
    runner = CliRunner()
    result = runner.invoke(main, ["version"])
    assert result.exit_code == 0
    assert result.output.strip().count(".") == 2  # x.y.z
```

- [ ] **Step 5: Run lint, typecheck, tests — all green**

```bash
make lint && make typecheck && make test
```

Expected: ruff PASS · pyright PASS (0 errors) · pytest 1 passed.

- [ ] **Step 6: Commit**

```bash
git add projects/VRM-model/src projects/VRM-model/tests
git commit -m "vrm: stub package layout with CLI entrypoint and smoke test"
```

---

### Task 4: Smoke script for base inference (placeholder, no GPU yet)

**Files:**
- Create: `projects/VRM-model/scripts/smoke-base-inference.sh`

- [ ] **Step 1: Write the smoke script (defers to in-pod execution; no GPU locally)**

`projects/VRM-model/scripts/smoke-base-inference.sh`:

```bash
#!/usr/bin/env bash
# Smoke-test base Qwen2.5-VL-7B-Instruct inference.
# Locally this only verifies the model can be loaded from HF; full inference runs in-pod.
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[smoke] No GPU detected — verifying model card downloads from HF (no inference)."
    uv run python -c "
from huggingface_hub import HfApi
api = HfApi()
info = api.model_info('Qwen/Qwen2.5-VL-7B-Instruct')
print(f'OK: model={info.modelId} sha={info.sha[:8]} downloads={info.downloads}')
"
    exit 0
fi

# In-GPU smoke: actual reference inference. Requires train extras installed.
uv run --with-requirements pyproject.toml --extra train python -c "
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

print('CUDA available:', torch.cuda.is_available(), 'device count:', torch.cuda.device_count())
model_id = 'Qwen/Qwen2.5-VL-7B-Instruct'
proc = AutoProcessor.from_pretrained(model_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='cuda:0')
print('Loaded:', model.config.model_type, '| hidden:', model.config.hidden_size)
print('Smoke OK.')
"
```

- [ ] **Step 2: Make it executable and test the no-GPU path**

```bash
chmod +x projects/VRM-model/scripts/smoke-base-inference.sh
make smoke
```

Expected (on dev laptop without GPU): `[smoke] No GPU detected ...` then `OK: model=Qwen/Qwen2.5-VL-7B-Instruct ...`.

- [ ] **Step 3: Commit**

```bash
git add projects/VRM-model/scripts/smoke-base-inference.sh
git commit -m "vrm: add base inference smoke script (CPU model-card check + GPU full check)"
```

---

## Done when

- [ ] `make sync && make lint && make typecheck && make test && make smoke` all green locally.
- [ ] `git status` shows clean working tree.
- [ ] Sub-plan 03 (data schema) and 02 (runpod infra) can begin (their first tasks add files into the now-existing package skeleton).
