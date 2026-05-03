# VRM-7B bootstrap (zero-to-first-training)

This is the exact one-time sequence from fresh clone to first Stage 1 SFT
launch. Pair with [`runbook.md`](runbook.md) for ongoing operations and
[`budgets.md`](budgets.md) for cost envelope.

Takes ~45 min end-to-end once you have all credentials in hand.

---

## 0. Prerequisites you need before running anything

Get these tokens first; most can be created in 2-5 min each:

| Credential | Where | Scope needed |
|---|---|---|
| `RUNPOD_API_KEY` | <https://www.runpod.io/console/user/settings> (API Keys) | Write |
| `HF_TOKEN` | <https://huggingface.co/settings/tokens> | Write on `tech-sumit` org |
| `WANDB_API_KEY` | <https://wandb.ai/authorize> | default |
| `ANTHROPIC_API_KEY` | <https://console.anthropic.com/settings/keys> | default |
| `OPENAI_API_KEY` | <https://platform.openai.com/api-keys> | default |
| `GH_TOKEN_FOR_DISPATCH` | <https://github.com/settings/tokens> (classic PAT) | `repo` scope |
| `SLACK_WEBHOOK_VRM` *(optional)* | Slack incoming-webhook integration | n/a |

Also ensure:
- `gh` CLI installed + authenticated (`gh auth status`)
- `uv` installed (`uv --version`, else `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker (only if you want local image builds) — CI builds them for you on push

---

## 1. Confirm you're on `main` and code is green

```bash
cd projects/VRM-model
make sync && make lint && make typecheck && make test
```

Expected: lint pass · pyright 0 errors (6 expected CUDA-only warnings) · 113 tests pass.

---

## 2. Create the RunPod network volume (one-time, ~$140/month standing cost)

```bash
export RUNPOD_API_KEY=rp_...
uv run python scripts/bootstrap-runpod-volume.py \
    --name vrm-7b-volume --size-gb 2000 --region US-GA-2
```

Prints the volume ID. Copy it — you'll paste it into GH variable
`VRM_NETWORK_VOLUME_ID` in the next step.

---

## 3. Verify HuggingFace Hub access (no cost)

```bash
export HF_TOKEN=hf_...
uv run python scripts/bootstrap-hf.py --org tech-sumit
# Optionally pre-create placeholder repos:
uv run python scripts/bootstrap-hf.py --org tech-sumit --create-repos
```

Expected: `OK`. If fails: fix token scope or org membership before proceeding.

---

## 4. Set GitHub repo secrets + variables

Export all required values into your current shell env, then run:

```bash
export RUNPOD_API_KEY=rp_...
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GH_TOKEN_FOR_DISPATCH=ghp_...
export SLACK_WEBHOOK_VRM=https://hooks.slack.com/...   # optional
export VRM_NETWORK_VOLUME_ID=vol_xxxxxxxx               # from step 2

bash scripts/bootstrap-gh.sh
# Or interactively (prompts for each missing value):
bash scripts/bootstrap-gh.sh --interactive
# Dry-run first if unsure:
bash scripts/bootstrap-gh.sh --dry-run
```

Verify:

```bash
gh secret list
gh variable list
```

You should see 6-7 secrets and 16 variables set.

---

## 5. Push to `main` so CI builds + publishes Docker images to GHCR

```bash
# Back at repo root
cd ../..
git push origin main
```

Watch `vrm-ci.yml`:

```bash
gh run watch $(gh run list --workflow=vrm-ci.yml --limit=1 --json databaseId -q '.[0].databaseId')
```

On success, three images appear at `ghcr.io/tech-sumit/vrm-{train,eval,dataprep}:latest`.

---

## 6. Launch the data build (first real cost: ~$500)

```bash
gh workflow run vrm-data-build.yml \
    -f data_version=v1-$(date +%Y-%m-%d) \
    -f include_distillation=true
```

This launches a CPU-only RunPod pod (no H200 cost — teacher APIs dominate).
Monitor:

```bash
gh run watch $(gh run list --workflow=vrm-data-build.yml --limit=1 --json databaseId -q '.[0].databaseId')
```

Pod self-destroys on completion (~3 hours). Two HF datasets will be created
at `tech-sumit/vrm-sft-<version>` and `tech-sumit/vrm-rl-<version>`.

---

## 7. Green light: launch Stage 1 SFT

Once step 6 completes (check HF for the datasets), launch SFT:

```bash
gh workflow run vrm-train-sft.yml \
    -f data_version=v1-YYYY-MM-DD \
    -f run_name=sft-$(date +%Y-%m-%d) \
    -f mode=full
```

Expected duration: 18-30h on 8×H200. Cost: ~$600-1,000.

Webhook `vrm-sft-completed` will auto-trigger `vrm-eval.yml` (quick suite) when done.

---

## Troubleshooting bootstrap

| Symptom | Fix |
|---|---|
| `scripts/bootstrap-gh.sh` says `ERROR: gh not authenticated` | `gh auth login` |
| `bootstrap-runpod-volume.py` 401 | Re-check `RUNPOD_API_KEY` (no `Bearer` prefix needed) |
| `bootstrap-hf.py` "not a member of 'tech-sumit'" | Get added to org or use `--org <your-username>` |
| CI fails with `denied: permission_denied` pushing to GHCR | Enable "Read and write permissions" in repo Settings → Actions → General → Workflow permissions |
| `vrm-data-build` stuck in queue | Check RunPod console capacity in `VRM_REGION` |

See [`runbook.md`](runbook.md) for ongoing operations.
