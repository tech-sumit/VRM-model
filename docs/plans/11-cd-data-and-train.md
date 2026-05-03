# Sub-plan 11 — CD workflows: data-build, train-sft, train-grpo

> Index: [`../plan.md`](../plan.md) · Depends on: [02 runpod infra](02-runpod-infra.md) · [05 filter+distill](05-data-filter-distill.md) · [06 SFT](06-stage1-sft.md) · [07 GRPO](07-stage2-grpo.md) · [10 CI](10-ci-workflow.md) · Status: ☐

**Goal:** Three `workflow_dispatch` workflows that orchestrate RunPod pods for the heavy compute steps. Each follows the same pattern: GH Actions kicks off the pod via `vrm.infra.runpod`, the pod runs the actual workload via `pod-entrypoint.sh`, and the pod posts back via `repository_dispatch` on completion (handled in sub-plan 12).

**Architecture:** All three workflows are thin — they install uv, set GH-Actions secrets as env vars, call `python -m vrm.infra.runpod launch-{train,eval,dataprep}`, and exit. The pod self-runs for hours/days; GH Actions does NOT wait.

**Tech Stack:** GitHub Actions · `astral-sh/setup-uv@v3` · stored repo secrets.

---

### Task 1: Required repo secrets (one-time setup)

- [ ] **Step 1: Add secrets via `gh secret set`**

```bash
gh secret set RUNPOD_API_KEY -a actions
gh secret set HF_TOKEN -a actions
gh secret set ANTHROPIC_API_KEY -a actions
gh secret set OPENAI_API_KEY -a actions
gh secret set WANDB_API_KEY -a actions
gh secret set SLACK_WEBHOOK_VRM -a actions    # optional
gh secret set GH_TOKEN_FOR_DISPATCH -a actions # PAT with repo:dispatch scope
```

- [ ] **Step 2: Add repo variables (non-secret config)**

```bash
gh variable set VRM_TRAIN_IMAGE --body "ghcr.io/tech-sumit/vrm-train:latest"
gh variable set VRM_EVAL_IMAGE --body "ghcr.io/tech-sumit/vrm-eval:latest"
gh variable set VRM_DATAPREP_IMAGE --body "ghcr.io/tech-sumit/vrm-dataprep:latest"
gh variable set VRM_GPU_TYPE_TRAIN --body "NVIDIA H200"
gh variable set VRM_GPU_COUNT_TRAIN --body "8"
gh variable set VRM_GPU_TYPE_EVAL --body "NVIDIA H200"
gh variable set VRM_GPU_COUNT_EVAL --body "1"
gh variable set VRM_REGION --body "US-GA-2"
gh variable set VRM_NETWORK_VOLUME_ID --body "<paste_volume_id_from_runpod_console>"
gh variable set HF_ORG --body "tech-sumit"
gh variable set GH_REPO --body "tech-sumit/react-webmcp"
```

- [ ] **Step 3: Verify**

```bash
gh secret list
gh variable list
```

---

### Task 2: `vrm-data-build.yml`

**Files:**
- Create: `.github/workflows/vrm-data-build.yml`

- [ ] **Step 1: Write the workflow**

`.github/workflows/vrm-data-build.yml`:

```yaml
name: vrm-data-build

on:
  workflow_dispatch:
    inputs:
      data_version:
        description: "Output data version tag (e.g. v1, v2-test)"
        required: true
        type: string
      include_distillation:
        description: "Run teacher distillation (Claude+GPT-4o)? Otherwise just normalize+filter"
        required: true
        type: boolean
        default: true

concurrency:
  group: vrm-data-${{ inputs.data_version }}
  cancel-in-progress: false

defaults:
  run:
    working-directory: projects/VRM-model

jobs:
  launch-pod:
    name: launch dataprep pod
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
          cache-dependency-glob: projects/VRM-model/uv.lock
      - run: uv python install 3.11
      - run: uv sync --extra dev

      - name: Launch RunPod dataprep pod
        env:
          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_ORG: ${{ vars.HF_ORG }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          SLACK_WEBHOOK_VRM: ${{ secrets.SLACK_WEBHOOK_VRM }}
          GH_REPO: ${{ vars.GH_REPO }}
          GH_TOKEN_FOR_DISPATCH: ${{ secrets.GH_TOKEN_FOR_DISPATCH }}
          VRM_DATAPREP_IMAGE: ${{ vars.VRM_DATAPREP_IMAGE }}
          VRM_REGION: ${{ vars.VRM_REGION }}
          VRM_NETWORK_VOLUME_ID: ${{ vars.VRM_NETWORK_VOLUME_ID }}
          VRM_MAX_USD_DATAPREP: "500"
          VRM_GIT_REPO: "https://github.com/${{ vars.GH_REPO }}.git"
          VRM_GIT_REF: ${{ github.sha }}
        run: |
          POD_ID=$(uv run vrm runpod launch-dataprep \
              --recipe configs/data/sft_recipe.yaml \
              --recipe configs/data/rl_recipe.yaml \
              --data-version "${{ inputs.data_version }}")
          echo "POD_ID=$POD_ID" >> $GITHUB_OUTPUT
          echo "Launched pod $POD_ID"
        id: launch

      - name: Comment summary
        if: always()
        run: |
          echo "### Dataprep pod launched" >> $GITHUB_STEP_SUMMARY
          echo "- pod_id: \`${{ steps.launch.outputs.POD_ID }}\`" >> $GITHUB_STEP_SUMMARY
          echo "- data_version: \`${{ inputs.data_version }}\`" >> $GITHUB_STEP_SUMMARY
          echo "- distillation: \`${{ inputs.include_distillation }}\`" >> $GITHUB_STEP_SUMMARY
          echo "- monitor: https://www.runpod.io/console/pods/${{ steps.launch.outputs.POD_ID }}" >> $GITHUB_STEP_SUMMARY
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/vrm-data-build.yml
git commit -m "ci: add vrm-data-build workflow (kicks off dataprep pod via RunPod API)"
```

- [ ] **Step 3: Smoke**

```bash
gh workflow run vrm-data-build.yml -f data_version=smoke -f include_distillation=false
gh run watch
```

Expected: workflow exits in ~30s with the pod ID in summary.

---

### Task 3: `vrm-train-sft.yml`

**Files:**
- Create: `.github/workflows/vrm-train-sft.yml`

- [ ] **Step 1: Write the workflow**

`.github/workflows/vrm-train-sft.yml`:

```yaml
name: vrm-train-sft

on:
  workflow_dispatch:
    inputs:
      data_version:
        description: "SFT data version (must exist as tech-sumit/vrm-7b-sft-{version})"
        required: true
        type: string
      run_name:
        description: "Run name (e.g. sft-2026-05-03)"
        required: true
        type: string
      mode:
        description: "Full FT or LoRA"
        required: true
        type: choice
        options: [full, lora]
        default: full

concurrency:
  group: vrm-train-sft-${{ inputs.run_name }}
  cancel-in-progress: false

defaults:
  run:
    working-directory: projects/VRM-model

jobs:
  launch-pod:
    name: launch SFT pod
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
          cache-dependency-glob: projects/VRM-model/uv.lock
      - run: uv python install 3.11
      - run: uv sync --extra dev

      - name: Resolve config path
        id: cfg
        run: |
          if [ "${{ inputs.mode }}" = "lora" ]; then
            echo "path=configs/stage1_sft_lora.yaml" >> $GITHUB_OUTPUT
          else
            echo "path=configs/stage1_sft_full.yaml" >> $GITHUB_OUTPUT
          fi

      - name: Launch RunPod SFT pod
        env:
          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_ORG: ${{ vars.HF_ORG }}
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
          WANDB_PROJECT: vrm-7b
          WANDB_ENTITY: ${{ vars.HF_ORG }}
          SLACK_WEBHOOK_VRM: ${{ secrets.SLACK_WEBHOOK_VRM }}
          GH_REPO: ${{ vars.GH_REPO }}
          GH_TOKEN_FOR_DISPATCH: ${{ secrets.GH_TOKEN_FOR_DISPATCH }}
          VRM_TRAIN_IMAGE: ${{ vars.VRM_TRAIN_IMAGE }}
          VRM_GPU_TYPE_TRAIN: ${{ vars.VRM_GPU_TYPE_TRAIN }}
          VRM_GPU_COUNT_TRAIN: ${{ vars.VRM_GPU_COUNT_TRAIN }}
          VRM_REGION: ${{ vars.VRM_REGION }}
          VRM_NETWORK_VOLUME_ID: ${{ vars.VRM_NETWORK_VOLUME_ID }}
          VRM_MAX_USD_SFT: "1500"
          VRM_GIT_REPO: "https://github.com/${{ vars.GH_REPO }}.git"
          VRM_GIT_REF: ${{ github.sha }}
        run: |
          POD_ID=$(uv run vrm runpod launch-train \
              --stage sft \
              --config "${{ steps.cfg.outputs.path }}" \
              --data-version "${{ inputs.data_version }}" \
              --run-name "${{ inputs.run_name }}")
          echo "POD_ID=$POD_ID" >> $GITHUB_OUTPUT
        id: launch

      - name: Summary
        if: always()
        run: |
          echo "### SFT pod launched" >> $GITHUB_STEP_SUMMARY
          echo "- run_name: \`${{ inputs.run_name }}\`" >> $GITHUB_STEP_SUMMARY
          echo "- pod_id: \`${{ steps.launch.outputs.POD_ID }}\`" >> $GITHUB_STEP_SUMMARY
          echo "- mode: \`${{ inputs.mode }}\`" >> $GITHUB_STEP_SUMMARY
          echo "- expected wall time: ~24h on 8×H200" >> $GITHUB_STEP_SUMMARY
          echo "- monitor: https://wandb.ai/${{ vars.HF_ORG }}/vrm-7b" >> $GITHUB_STEP_SUMMARY
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/vrm-train-sft.yml
git commit -m "ci: add vrm-train-sft workflow (launches Stage 1 SFT pod)"
```

---

### Task 4: `vrm-train-grpo.yml`

**Files:**
- Create: `.github/workflows/vrm-train-grpo.yml`

- [ ] **Step 1: Write the workflow**

`.github/workflows/vrm-train-grpo.yml`:

```yaml
name: vrm-train-grpo

on:
  workflow_dispatch:
    inputs:
      sft_checkpoint:
        description: "HF repo of the post-SFT checkpoint (e.g. tech-sumit/vrm-7b-sft-2026-05-03)"
        required: true
        type: string
      data_version:
        description: "RL data version (must exist as tech-sumit/vrm-7b-rl-{version})"
        required: true
        type: string
      run_name:
        description: "Run name (e.g. grpo-2026-05-04)"
        required: true
        type: string

concurrency:
  group: vrm-train-grpo-${{ inputs.run_name }}
  cancel-in-progress: false

defaults:
  run:
    working-directory: projects/VRM-model

jobs:
  launch-pod:
    name: launch GRPO pod
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
          cache-dependency-glob: projects/VRM-model/uv.lock
      - run: uv python install 3.11
      - run: uv sync --extra dev

      - name: Launch RunPod GRPO pod
        env:
          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_ORG: ${{ vars.HF_ORG }}
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
          WANDB_PROJECT: vrm-7b
          WANDB_ENTITY: ${{ vars.HF_ORG }}
          SLACK_WEBHOOK_VRM: ${{ secrets.SLACK_WEBHOOK_VRM }}
          GH_REPO: ${{ vars.GH_REPO }}
          GH_TOKEN_FOR_DISPATCH: ${{ secrets.GH_TOKEN_FOR_DISPATCH }}
          VRM_TRAIN_IMAGE: ${{ vars.VRM_TRAIN_IMAGE }}
          VRM_GPU_TYPE_TRAIN: ${{ vars.VRM_GPU_TYPE_TRAIN }}
          VRM_GPU_COUNT_TRAIN: ${{ vars.VRM_GPU_COUNT_TRAIN }}
          VRM_REGION: ${{ vars.VRM_REGION }}
          VRM_NETWORK_VOLUME_ID: ${{ vars.VRM_NETWORK_VOLUME_ID }}
          VRM_MAX_USD_GRPO: "8000"
          VRM_GIT_REPO: "https://github.com/${{ vars.GH_REPO }}.git"
          VRM_GIT_REF: ${{ github.sha }}
        run: |
          POD_ID=$(uv run vrm runpod launch-train \
              --stage grpo \
              --config configs/stage2_grpo.yaml \
              --sft-checkpoint "${{ inputs.sft_checkpoint }}" \
              --data-version "${{ inputs.data_version }}" \
              --run-name "${{ inputs.run_name }}")
          echo "POD_ID=$POD_ID" >> $GITHUB_OUTPUT
        id: launch

      - name: Summary
        if: always()
        run: |
          echo "### GRPO pod launched" >> $GITHUB_STEP_SUMMARY
          echo "- run_name: \`${{ inputs.run_name }}\`" >> $GITHUB_STEP_SUMMARY
          echo "- sft_checkpoint: \`${{ inputs.sft_checkpoint }}\`" >> $GITHUB_STEP_SUMMARY
          echo "- pod_id: \`${{ steps.launch.outputs.POD_ID }}\`" >> $GITHUB_STEP_SUMMARY
          echo "- expected wall time: 10-14 days on 8×H200" >> $GITHUB_STEP_SUMMARY
          echo "- budget tripwire: \$8,000 (pod self-destroys if exceeded)" >> $GITHUB_STEP_SUMMARY
          echo "- monitor: https://wandb.ai/${{ vars.HF_ORG }}/vrm-7b" >> $GITHUB_STEP_SUMMARY
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/vrm-train-grpo.yml
git commit -m "ci: add vrm-train-grpo workflow (launches Stage 2 GRPO pod)"
```

---

### Task 5: Test the dispatch flow with a tiny dry-run pod

- [ ] **Step 1: Use a CPU pod with `VRM_TASK=dataprep` to validate the launch path without GPU spend**

```bash
gh workflow run vrm-data-build.yml -f data_version=ci-smoke -f include_distillation=false
gh run list --workflow=vrm-data-build.yml --limit 1
gh run view --log
```

Expected: workflow finishes in 30-60s with a pod ID. Manually destroy the pod afterwards via `uv run vrm runpod destroy <id>` (or the pod self-completes once distillation is skipped).

- [ ] **Step 2: Capture observations in `docs/runbook.md`**

(Done in sub-plan 13.)

---

## Done when

- [ ] All three workflows are committed and trigger from the GitHub UI.
- [ ] At least one dispatch successfully creates a pod (visible in RunPod console).
- [ ] Pod logs show `[budget] burn=$X.XX/hr limit=$Y` from the budget daemon.
- [ ] Sub-plan 12 (eval + release) wires the `repository_dispatch` listener for completion events.
