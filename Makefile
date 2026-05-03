.PHONY: help sync lint fmt typecheck test test-integration smoke clean \
        image-train image-eval image-dataprep \
        data train-sft train-grpo train-rejection eval release

help: ## list targets
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

smoke: ## run base Qwen2.5-VL-7B reference inference (HF download check on CPU)
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

train-rejection: ## launch Stage 3 rejection-sampled SFT pod (requires GRPO_CHECKPOINT)
	uv run python -m vrm.infra.runpod launch-train \
	    --stage rejection --config configs/stage3_rejection_sft.yaml \
	    --sft-checkpoint $(GRPO_CHECKPOINT) \
	    --data-version $(DATA_VERSION) --run-name $(RUN_NAME)

SUITE ?= full
eval: ## launch eval pod for a checkpoint (requires CHECKPOINT)
	uv run python -m vrm.infra.runpod launch-eval \
	    --checkpoint $(CHECKPOINT) --suite $(SUITE)

release: ## tag-and-release flow (manual git tag triggers vrm-release.yml)
	@echo "git tag -a vrm-7b-vX.Y.Z -m 'source=tech-sumit/vrm-7b-grpo-...' && git push origin vrm-7b-vX.Y.Z"

# --- One-time bootstrap (see docs/bootstrap.md) ---
HF_ORG ?= tech-sumit

bootstrap-gh: ## set GH secrets+variables from env (pass INTERACTIVE=1 to prompt)
	bash scripts/bootstrap-gh.sh $(if $(INTERACTIVE),--interactive,)

bootstrap-runpod: ## create 2TB RunPod network volume (requires RUNPOD_API_KEY)
	@uv run python scripts/bootstrap-runpod-volume.py

bootstrap-hf: ## verify HF Hub access (requires HF_TOKEN)
	@uv run python scripts/bootstrap-hf.py --org $(HF_ORG)

bootstrap: bootstrap-hf bootstrap-runpod bootstrap-gh ## full bootstrap: HF + RunPod volume + GH secrets/vars

data-build: ## trigger vrm-data-build workflow (requires DATA_VERSION, ~3h, ~$500)
	gh workflow run vrm-data-build.yml \
	    -f data_version=$(DATA_VERSION) \
	    -f include_distillation=$(if $(SKIP_DISTILL),false,true)
