# VRM-7B Operations Runbook

End-to-end ops manual: from cold start to public release. Pair with
[`docs/budgets.md`](budgets.md) for cost rates and tripwires.

## Roles

| Role | Action |
|---|---|
| Maintainer | Land PRs, trigger workflows, tag releases |
| Operator | Watch RunPod console + Grafana, kill runaway pods |
| Reviewer | Approve `vrm-release` workflow runs |

## 0. One-time setup

1. **GitHub repo secrets** (`Settings → Secrets and variables → Actions → Secrets`):
   - `RUNPOD_API_KEY`
   - `HF_TOKEN` (write scope on `tech-sumit` org)
   - `WANDB_API_KEY`
   - `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (only needed for `vrm-data-build`)
   - `GH_TOKEN_FOR_DISPATCH` (PAT with `repo:dispatch` scope)
   - `SLACK_WEBHOOK_VRM` (optional)

2. **GitHub repo variables** (`...→ Variables`):
   - `HF_ORG=tech-sumit`
   - `WANDB_PROJECT=vrm-7b`, `WANDB_ENTITY=tech-sumit`
   - `VRM_TRAIN_IMAGE=ghcr.io/tech-sumit/vrm-train:latest`
   - `VRM_EVAL_IMAGE=ghcr.io/tech-sumit/vrm-eval:latest`
   - `VRM_DATAPREP_IMAGE=ghcr.io/tech-sumit/vrm-dataprep:latest`
   - `VRM_NETWORK_VOLUME_ID` (create network volume in RunPod console first)
   - `VRM_REGION=US-GA-2` (or matching the volume)
   - `VRM_GPU_TYPE_TRAIN=NVIDIA H200`
   - `VRM_GPU_COUNT_TRAIN=8`
   - `VRM_GPU_TYPE_EVAL=NVIDIA H200`
   - `VRM_GPU_COUNT_EVAL=1`
   - `VRM_MAX_USD_DATAPREP=500`, `VRM_MAX_USD_SFT=1500`,
     `VRM_MAX_USD_GRPO=8000`, `VRM_MAX_USD_EVAL=200`

3. **Local dev shell**:
   ```bash
   cd projects/VRM-model
   cp .env.example .env  # fill in tokens
   make sync && make test
   ```

## 1. Build dataset shards (cold start)

```bash
gh workflow run vrm-data-build.yml -f data_version=v1
```

What happens:
- GH Actions launches a CPU-only RunPod pod
- Pod downloads sources (HF), runs `pass@8` filter using vLLM on a side GPU
  (small Qwen2.5-VL via `--limit` for speed), then teacher-distills with
  Claude+GPT-4o
- Final shards uploaded to `tech-sumit/vrm-7b-sft-v1` and
  `tech-sumit/vrm-7b-rl-v1` on HF Hub
- Pod self-destroys; webhook fires `vrm-dataprep-completed`

Expected cost: ~$300-500 (mostly teacher API).

## 2. Stage 1 SFT

```bash
gh workflow run vrm-train-sft.yml \
    -f data_version=v1 \
    -f run_name=sft-2026-05-03 \
    -f mode=full
```

What happens:
- 8×H200 pod boots, pulls HEAD, runs LLaMA-Factory (3 epochs)
- Checkpoints to network volume; final pushed to
  `tech-sumit/vrm-7b-sft-2026-05-03`
- Webhook `vrm-sft-completed` triggers `vrm-eval` (full suite, baseline check)

Expected duration: 18-30h. Cost: ~$1,000-1,400.

## 3. Stage 2 GRPO

```bash
gh workflow run vrm-train-grpo.yml \
    -f sft_checkpoint=tech-sumit/vrm-7b-sft-2026-05-03 \
    -f data_version=v1 \
    -f run_name=grpo-2026-05-04
```

What happens:
- 8×H200 pod boots, pulls HEAD, runs TRL `GRPOTrainer` (4000 steps)
- Reward = 0.1·format + 0.9·accuracy from deterministic verifiers
- vLLM rolls out 16 generations per prompt (DAPO settings: clip-higher,
  dynamic sampling, no KL)
- Checkpoints every 200 steps to volume + every 1000 to HF Hub
- Final: `tech-sumit/vrm-7b-grpo-2026-05-04`; webhook fires `vrm-grpo-completed`

Expected duration: 10-14d. Cost: ~$7,500-8,000.

### Watching it
- Grafana dashboard `vrm` (provisioned via `dashboards-push`)
- W&B run page (linked in Slack on start)
- `runpod pod logs <pod_id>` from RunPod console

### When to abort
- Format reward < 0.5 by step 100 → wrong template
- Accuracy reward stuck at 0 → reward function bug or data corrupt
- Loss explodes → clip thresholds wrong
- Budget tripwire fires → check `vrm.infra.budget` logs

## 4. Eval

Auto-runs after each train completes, or trigger manually:
```bash
gh workflow run vrm-eval.yml \
    -f checkpoint=tech-sumit/vrm-7b-grpo-2026-05-04 \
    -f suite=full
```

Output: `report.md` in eval pod's volume, posted to Slack and as PR comment
on the originating training run (if applicable).

## 5. Release

Annotated tag triggers `vrm-release.yml`:
```bash
git tag -a vrm-7b-v1.0.0 -m "source=tech-sumit/vrm-7b-grpo-2026-05-04"
git push origin vrm-7b-v1.0.0
```

The workflow:
- Promotes private weights to public `tech-sumit/vrm-7b`
- Creates GitHub Release with model card + eval delta

## Troubleshooting

### Pod won't start
1. RunPod console → check capacity in `VRM_REGION`. If H200s are out, retry
   in 10-30min or switch region (and recreate the network volume there).
2. Check pod logs for image pull failures; verify GHCR image exists for the
   commit SHA referenced in `VRM_TRAIN_IMAGE`.

### vLLM OOM during GRPO rollouts
- Lower `vllm.gpu_memory_utilization` from 0.85 to 0.75
- Lower `grpo.num_generations` from 16 to 8
- Lower `generation.max_new_tokens` from 8192 to 4096

### Reward stuck at 0
- Verify dataset on HF Hub: `huggingface-cli download tech-sumit/vrm-7b-rl-v1`
  and inspect a few records — answer/verifier fields must be populated
- Run reward fn locally: `uv run python -c "from vrm.train.reward import ..."`

### Budget tripwire fired but pod still running
- Check `RUNPOD_API_KEY` is set in pod env (so `_self_destruct` can call
  destroy)
- Manual fallback: `runpod pod stop <pod_id>` from RunPod console

### Need to resume after pod death
- Find latest checkpoint on HF Hub: `huggingface-cli scan-cache | grep vrm-7b`
- Re-launch with that as `--sft-checkpoint` (for GRPO) or
  `--data-dir-override` for SFT data

## Releasing model card updates

```bash
huggingface-cli repo-files-update tech-sumit/vrm-7b README.md
```
