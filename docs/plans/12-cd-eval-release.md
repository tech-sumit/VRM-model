# Sub-plan 12 — CD workflows: eval, release

> Index: [`../plan.md`](../plan.md) · Depends on: [09 eval](09-eval-harness.md) · [11 CD train](11-cd-data-and-train.md) · Status: ☐

**Goal:** Two more workflows that close the loop:
1. `vrm-eval.yml` — runs eval on a checkpoint. Triggered manually or by `repository_dispatch` (`vrm-sft-completed`, `vrm-grpo-completed`, `vrm-rejection-completed`) from in-pod webhooks.
2. `vrm-release.yml` — on git tag `vrm-7b-v*.*.*`, copies weights from the internal HF repo to a public org repo, generates the technical-report markdown from the latest eval JSON, attaches everything to a GitHub Release.

**Architecture:** `vrm-eval.yml` launches a 1×H200 eval pod via `vrm.infra.runpod launch-eval`, then exits. The eval pod runs VLMEvalKit, parses results, posts a `vrm-eval-completed` dispatch with the markdown report attached as a JSON payload field. `vrm-release.yml` is a single GH Actions job that runs entirely on `ubuntu-latest`: clones HF repos, copies weights via the HF Hub Python API, calls `vrm.eval.compare` to render delta tables, and uses `gh release create`.

**Tech Stack:** GitHub Actions · `huggingface_hub` Python API · `gh release`.

---

### Task 1: `vrm-eval.yml`

**Files:**
- Create: `.github/workflows/vrm-eval.yml`

- [ ] **Step 1: Write the workflow (manual + dispatch triggers)**

`.github/workflows/vrm-eval.yml`:

```yaml
name: vrm-eval

on:
  workflow_dispatch:
    inputs:
      checkpoint:
        description: "HF model repo to eval (e.g. tech-sumit/vrm-7b-grpo-2026-05-04)"
        required: true
        type: string
      suite:
        description: "Eval suite name"
        required: true
        type: choice
        options: [full, quick, negative_control]
        default: full
  repository_dispatch:
    types:
      - vrm-sft-completed
      - vrm-grpo-completed
      - vrm-rejection-completed

concurrency:
  group: vrm-eval-${{ github.event.inputs.checkpoint || github.event.client_payload.hf_repo }}
  cancel-in-progress: false

defaults:
  run:
    working-directory: projects/VRM-model

jobs:
  resolve:
    name: resolve checkpoint + suite
    runs-on: ubuntu-latest
    timeout-minutes: 2
    outputs:
      checkpoint: ${{ steps.r.outputs.checkpoint }}
      suite: ${{ steps.r.outputs.suite }}
      run_name: ${{ steps.r.outputs.run_name }}
    steps:
      - id: r
        run: |
          if [ -n "${{ github.event.inputs.checkpoint }}" ]; then
            echo "checkpoint=${{ github.event.inputs.checkpoint }}" >> $GITHUB_OUTPUT
            echo "suite=${{ github.event.inputs.suite }}" >> $GITHUB_OUTPUT
            echo "run_name=eval-${{ github.event.inputs.suite }}-$(date +%Y%m%d-%H%M%S)" >> $GITHUB_OUTPUT
          else
            CKPT="${{ github.event.client_payload.hf_repo }}"
            echo "checkpoint=$CKPT" >> $GITHUB_OUTPUT
            # Auto-suite by event source: full for completed RL, quick for completed SFT
            if [ "${{ github.event.action }}" = "vrm-grpo-completed" ]; then
              echo "suite=full" >> $GITHUB_OUTPUT
            else
              echo "suite=quick" >> $GITHUB_OUTPUT
            fi
            echo "run_name=eval-auto-$(date +%Y%m%d-%H%M%S)" >> $GITHUB_OUTPUT
          fi

  launch-pod:
    name: launch eval pod
    needs: resolve
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

      - name: Launch eval pod
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
          VRM_EVAL_IMAGE: ${{ vars.VRM_EVAL_IMAGE }}
          VRM_GPU_TYPE_EVAL: ${{ vars.VRM_GPU_TYPE_EVAL }}
          VRM_GPU_COUNT_EVAL: ${{ vars.VRM_GPU_COUNT_EVAL }}
          VRM_REGION: ${{ vars.VRM_REGION }}
          VRM_NETWORK_VOLUME_ID: ${{ vars.VRM_NETWORK_VOLUME_ID }}
          VRM_MAX_USD_EVAL: "200"
          VRM_GIT_REPO: "https://github.com/${{ vars.GH_REPO }}.git"
          VRM_GIT_REF: ${{ github.sha }}
        run: |
          POD_ID=$(uv run vrm runpod launch-eval \
              --checkpoint "${{ needs.resolve.outputs.checkpoint }}" \
              --suite "${{ needs.resolve.outputs.suite }}")
          echo "POD_ID=$POD_ID" >> $GITHUB_OUTPUT
        id: launch

      - name: Summary
        if: always()
        run: |
          echo "### Eval pod launched" >> $GITHUB_STEP_SUMMARY
          echo "- checkpoint: \`${{ needs.resolve.outputs.checkpoint }}\`" >> $GITHUB_STEP_SUMMARY
          echo "- suite: \`${{ needs.resolve.outputs.suite }}\`" >> $GITHUB_STEP_SUMMARY
          echo "- pod_id: \`${{ steps.launch.outputs.POD_ID }}\`" >> $GITHUB_STEP_SUMMARY
          echo "- expected wall time: 30-60min (quick) / 4-8h (full)" >> $GITHUB_STEP_SUMMARY

  comment-results:
    name: post results when eval completes
    if: github.event_name == 'repository_dispatch' && github.event.action == 'vrm-eval-completed'
    runs-on: ubuntu-latest
    timeout-minutes: 5
    permissions:
      issues: write
      pull-requests: write
    steps:
      - name: Format eval markdown comment
        run: |
          cat <<EOF >> $GITHUB_STEP_SUMMARY
          ## Eval completed: \`${{ github.event.client_payload.checkpoint }}\`
          Suite: \`${{ github.event.client_payload.suite }}\`

          See attached eval-summary.md in the run artifacts for full results.
          EOF
      - name: Save attached markdown to artifact
        run: |
          mkdir -p out
          echo "${{ toJSON(github.event.client_payload) }}" > out/eval-payload.json
      - uses: actions/upload-artifact@v4
        with:
          name: eval-${{ github.event.client_payload.run_name }}
          path: out/
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/vrm-eval.yml
git commit -m "ci: add vrm-eval workflow (manual + repository_dispatch from training pods)"
```

---

### Task 2: `vrm-release.yml`

**Files:**
- Create: `.github/workflows/vrm-release.yml`
- Create: `projects/VRM-model/scripts/release-promote.py`

- [ ] **Step 1: Write the release-promote helper**

`projects/VRM-model/scripts/release-promote.py`:

```python
"""Copy weights from a private internal HF repo to the public release repo.

Invoked by .github/workflows/vrm-release.yml on a `vrm-7b-v*.*.*` tag.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-repo", required=True, help="Private repo, e.g. tech-sumit/vrm-7b-grpo-2026-05-15")
    p.add_argument("--target-repo", required=True, help="Public repo, e.g. tech-sumit/VRM-7B")
    p.add_argument("--tag", required=True, help="Release tag, e.g. vrm-7b-v1.0.0")
    args = p.parse_args()

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(args.target_repo, repo_type="model", private=False, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        local = snapshot_download(args.source_repo, repo_type="model", local_dir=tmp, token=token)
        api.upload_folder(folder_path=local, repo_id=args.target_repo, repo_type="model",
                          commit_message=f"release: {args.tag} from {args.source_repo}")
        api.create_tag(args.target_repo, tag=args.tag, repo_type="model", token=token)

    print(f"Promoted {args.source_repo} → https://huggingface.co/{args.target_repo} @ {args.tag}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the workflow**

`.github/workflows/vrm-release.yml`:

```yaml
name: vrm-release

on:
  push:
    tags:
      - "vrm-7b-v*.*.*"

defaults:
  run:
    working-directory: projects/VRM-model

jobs:
  release:
    name: promote weights + create GH Release
    runs-on: ubuntu-latest
    timeout-minutes: 60
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
          cache-dependency-glob: projects/VRM-model/uv.lock
      - run: uv python install 3.11
      - run: uv sync --extra dev

      - name: Resolve source repo from tag annotation
        id: resolve
        run: |
          # Tag must be annotated with the source repo path, e.g.:
          #   git tag -a vrm-7b-v1.0.0 -m "source=tech-sumit/vrm-7b-grpo-2026-05-15"
          SRC=$(git tag -l --format='%(contents:body)' "${GITHUB_REF#refs/tags/}" | grep -oE 'source=[^ ]+' | head -1 | cut -d= -f2)
          if [ -z "$SRC" ]; then
            echo "::error::Tag must include 'source=<hf-repo>' in its annotation message"; exit 1
          fi
          echo "source_repo=$SRC" >> $GITHUB_OUTPUT
          echo "tag=${GITHUB_REF#refs/tags/}" >> $GITHUB_OUTPUT

      - name: Promote weights to public HF repo
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          uv run python scripts/release-promote.py \
              --source-repo "${{ steps.resolve.outputs.source_repo }}" \
              --target-repo "${{ vars.HF_ORG }}/VRM-7B" \
              --tag "${{ steps.resolve.outputs.tag }}"

      - name: Generate eval delta report
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          # Pull the latest eval-summary.md from the source repo (uploaded by the eval pod).
          # If absent, generate from local files included in the release tag.
          uv run python -c "
          from huggingface_hub import hf_hub_download
          import shutil, os
          try:
              p = hf_hub_download('${{ steps.resolve.outputs.source_repo }}', 'eval-summary.md',
                                  repo_type='model', token=os.environ['HF_TOKEN'])
              shutil.copy(p, 'eval-summary.md')
          except Exception as e:
              with open('eval-summary.md', 'w') as f:
                  f.write(f'# Eval results unavailable\n\n{e}\n')
          "

      - name: Create GitHub Release
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          gh release create "${{ steps.resolve.outputs.tag }}" \
              --title "VRM-7B ${{ steps.resolve.outputs.tag }}" \
              --notes-file projects/VRM-model/eval-summary.md \
              projects/VRM-model/eval-summary.md \
              projects/VRM-model/VRM-7B_model_spec.md
        working-directory: ${{ github.workspace }}
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/vrm-release.yml projects/VRM-model/scripts/release-promote.py
git commit -m "ci: add vrm-release workflow (on tag → HF promote + GH Release)"
```

---

### Task 3: End-to-end smoke of the eval → release loop

- [ ] **Step 1: Manually dispatch eval against the smoke SFT checkpoint**

```bash
gh workflow run vrm-eval.yml \
    -f checkpoint=tech-sumit/vrm-7b-sft-smoke-XXX \
    -f suite=quick
gh run list --workflow=vrm-eval.yml --limit 1
```

- [ ] **Step 2: Once eval pod completes (~30-60min), confirm `repository_dispatch`**

In RunPod logs, you should see a successful `POST /repos/.../dispatches` with `event_type=vrm-eval-completed`. The `vrm-eval / comment-results` job in GH should then run.

- [ ] **Step 3: Cut a dry-run release tag**

```bash
git tag -a vrm-7b-v0.0.1-smoke -m "source=tech-sumit/vrm-7b-sft-smoke-XXX"
git push origin vrm-7b-v0.0.1-smoke
gh run list --workflow=vrm-release.yml --limit 1
```

Expected: workflow completes; `https://huggingface.co/tech-sumit/VRM-7B` exists with the smoke weights; GH Release page shows attached `eval-summary.md` and `VRM-7B_model_spec.md`.

- [ ] **Step 4: Delete the smoke release/tag/repo if desired**

```bash
gh release delete vrm-7b-v0.0.1-smoke --yes
git push --delete origin vrm-7b-v0.0.1-smoke
```

---

## Done when

- [ ] `vrm-eval.yml` triggers from manual dispatch and from `repository_dispatch`.
- [ ] An eval pod completes and GH receives the dispatch (verified in workflow run history).
- [ ] `vrm-release.yml` ran end-to-end on a smoke tag and produced a public HF repo + GH Release.
- [ ] All 6 GH Actions workflows (ci, data-build, train-sft, train-grpo, eval, release) are present at `.github/workflows/`.
