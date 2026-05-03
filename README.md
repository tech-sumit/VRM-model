# VRM-7B

Open-weights 7-8B Visual Reasoning Model. SFT cold-start + GRPO RL post-training on `Qwen/Qwen2.5-VL-7B-Instruct`.

## Documentation

- [`VRM-7B_model_spec.md`](VRM-7B_model_spec.md) — locked model spec
- [`docs/design.md`](docs/design.md) — toolchain design
- [`docs/plan.md`](docs/plan.md) — implementation plan (index of 14 sub-plans)
- [`docs/bootstrap.md`](docs/bootstrap.md) — **zero-to-first-training one-time setup**
- [`docs/runbook.md`](docs/runbook.md) — ongoing operations manual
- [`docs/budgets.md`](docs/budgets.md) — cost envelope and tripwires

## Quickstart (local dev)

```bash
git clone https://github.com/tech-sumit/VRM-model.git && cd VRM-model
cp .env.example .env       # fill in tokens
make sync                  # install deps (no GPU needed for dev)
make test                  # unit tests
make lint                  # ruff + format check
make typecheck             # pyright
```

## Quickstart (RunPod operations)

```bash
# 1. Build dataset shards (CPU pod + teacher API)
make data DATA_VERSION=v1

# 2. Stage 1 SFT (~24h on 8xH200)
make train-sft DATA_VERSION=v1 RUN_NAME=sft-2026-05-03

# 3. Stage 2 GRPO (~10-14d on 8xH200)
make train-grpo \
    SFT_CHECKPOINT=tech-sumit/vrm-7b-sft-2026-05-03 \
    DATA_VERSION=v1 \
    RUN_NAME=grpo-2026-05-04

# 4. Eval
make eval CHECKPOINT=tech-sumit/vrm-7b-grpo-2026-05-04 SUITE=full

# 5. Release (annotated tag triggers vrm-release.yml)
git tag -a vrm-7b-v1.0.0 -m "source=tech-sumit/vrm-7b-grpo-2026-05-04"
git push origin vrm-7b-v1.0.0
```

CI/CD via GitHub Actions: see `.github/workflows/vrm-*.yml`.

## License

Apache-2.0.
