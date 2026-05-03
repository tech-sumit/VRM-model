# Sub-plan 13 — Runbook + budgets documentation

> Index: [`../plan.md`](../plan.md) · Depends on: all preceding sub-plans · Status: ☐

**Goal:** Two operational documents that let anyone (you in 6 months, a teammate, a contractor) run, monitor, and triage VRM-7B training/eval without re-reading the design + plan + spec.

**Architecture:** No code, just two markdown files. `runbook.md` is a step-by-step "how do I do X?" doc with copy-paste commands. `budgets.md` is a single-page table of cost rates + tripwires + escalation rules.

---

### Task 1: `docs/runbook.md`

**Files:**
- Create: `projects/VRM-model/docs/runbook.md`

- [ ] **Step 1: Write the runbook**

`projects/VRM-model/docs/runbook.md`:

```markdown
# VRM-7B Operations Runbook

> See [`design.md`](design.md) for *why* and [`plan.md`](plan.md) for *how it was built*. This file is the *how do I run it* manual.

## 0. Prerequisites (one-time per dev machine)

- macOS / Linux with Docker, `gh`, `git`, `uv`, `make` installed.
- `cp .env.example .env` and fill in:
  - `RUNPOD_API_KEY` (https://runpod.io/console/user/settings)
  - `HF_TOKEN` with `tech-sumit/` org write scope
  - `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (only for data-build)
  - `WANDB_API_KEY`
  - `VRM_NETWORK_VOLUME_ID` (created once via the RunPod console; see §5)
- `gh auth status` shows authenticated.

## 1. Local sanity

```bash
cd projects/VRM-model
make sync && make lint && make typecheck && make test
make smoke   # validates HF download of base model card (no GPU needed)
```

If any of these fail, do NOT proceed to remote operations.

## 2. Build a new dataset version

**When to run:** First time, or when source datasets / recipe / verifier logic changes.

```bash
gh workflow run vrm-data-build.yml \
    -f data_version=v$(date +%Y%m%d) \
    -f include_distillation=true
gh run watch
```

GH workflow exits in ~30s with a pod ID. The pod runs for **6-12 hours** and pushes results to:
- `huggingface.co/datasets/tech-sumit/vrm-7b-sft-v<date>`
- `huggingface.co/datasets/tech-sumit/vrm-7b-rl-v<date>`

Pod webhooks `vrm-dataprep-completed` to GH on success → no further action needed.
Cost: ~$500 (~$3-4K teacher-API spend on top, billed separately to Anthropic/OpenAI).

## 3. Train Stage 1 (SFT)

```bash
gh workflow run vrm-train-sft.yml \
    -f data_version=v<date> \
    -f run_name=sft-$(date +%Y%m%d) \
    -f mode=full
```

Wall time: **~24 hours** on 8×H200. Cost: **~$800** (full FT) or **~$400** (LoRA).
On completion, pod webhooks → kicks `vrm-eval.yml` with `suite=quick` automatically.
Manually check W&B dashboard at https://wandb.ai/tech-sumit/vrm-7b for `train/loss` curve.

**Acceptance:** loss decreases from ~3 to ~1.2; eval shows +5 to +10 MathVista points over base.

## 4. Train Stage 2 (GRPO)

```bash
gh workflow run vrm-train-grpo.yml \
    -f sft_checkpoint=tech-sumit/vrm-7b-sft-<date> \
    -f data_version=v<date> \
    -f run_name=grpo-$(date +%Y%m%d)
```

Wall time: **10-14 days** on 8×H200. Cost: **~$8000** (budget tripwire at $8K).
Pod auto-evals every 200 steps → uploads partial eval JSON to HF Hub.

**Watch for (W&B):**
- `train/reward` rising monotonically (with noise) → good.
- `train/reward` plateaus by step 800-1200 → done; consider stopping early.
- `train/completion_length` median climbing slowly → good.
- `train/completion_length` >7000 in p95 → overlong shaping kicking in; OK.
- KL >2.0 → bad; checkpoint may be drifting; raise `beta` to 0.005 next run.
- Reward saturating at 0.10 (only format) for >50 steps → reward hacking; check `MIN_THINK_TOKENS`.

## 5. Create the RunPod network volume (once)

Via RunPod console (no API equivalent yet):
1. Console → Storage → Network Volumes → Create.
2. Name: `vrm-data`. Size: 2 TB. Region: `US-GA-2` (must match `VRM_REGION`).
3. Copy the volume ID into `VRM_NETWORK_VOLUME_ID` (both `.env` and GH variable).

Cost: $140/month for 2 TB. Persists across all pods.

## 6. Eval a checkpoint

```bash
gh workflow run vrm-eval.yml \
    -f checkpoint=tech-sumit/vrm-7b-grpo-<date> \
    -f suite=full
```

Wall time: **4-8 hours**. Cost: ~$32-64.
Output: `vrm-eval-completed` dispatch with markdown report; download via:
```bash
gh run download $(gh run list --workflow=vrm-eval.yml --limit 1 --json databaseId -q '.[0].databaseId')
```

## 7. Release weights publicly

```bash
git tag -a vrm-7b-v1.0.0 -m "source=tech-sumit/vrm-7b-grpo-<date>"
git push origin vrm-7b-v1.0.0
```

`vrm-release.yml` fires on the tag → promotes weights to public `tech-sumit/VRM-7B` HF repo + creates GH Release with eval markdown attached.

Wall time: 30-60 min (mostly HF upload).
Cost: $0.

## 8. Common operations

### Manually destroy a pod
```bash
uv run vrm runpod destroy <pod_id>
# or via the RunPod console
```

### Resume a failed GRPO run
```bash
# 1. Identify the last checkpoint pushed to HF Hub:
huggingface-cli repo files-tree tech-sumit/vrm-7b-grpo-<date>
# 2. Set in env, re-launch:
gh workflow run vrm-train-grpo.yml \
    -f sft_checkpoint=tech-sumit/vrm-7b-grpo-<date>/checkpoint-<step> \
    -f data_version=v<date> \
    -f run_name=grpo-<date>-resume
```

### Rebuild Docker images manually (CI bypass)
```bash
make image-train image-eval image-dataprep
docker push ghcr.io/tech-sumit/vrm-train:dev
# update VRM_TRAIN_IMAGE to :dev for next pod launch
```

### Inspect pod logs without SSH
RunPod console → Pods → click pod → "Logs" tab. Or:
```bash
uv run vrm runpod status <pod_id>
```

### Trigger eval from a third-party checkpoint
```bash
gh workflow run vrm-eval.yml \
    -f checkpoint=Qwen/Qwen2.5-VL-7B-Instruct \
    -f suite=full
```

Lets you re-eval the base for comparison.

## 9. Triage: training run looks bad

| Symptom | Likely cause | Action |
|---|---|---|
| OOM at step 1 | model + optimizer state too big | Lower `per_device_train_batch_size` to 1, raise `gradient_accumulation_steps` |
| Loss NaN within first 50 steps | bf16 numerical issue | Check learning rate; usually too high |
| Reward stuck at 0.10 | reward hacking — model only learns format | Verify `MIN_THINK_TOKENS=50` in `vrm/data/verifiers/format.py`; raise to 100 |
| Reward stuck at 0.0 | format reward never fires | Check assistant prompt template; verify `<think>` tags are emitted |
| KL explodes (>5) | drift from reference model | Stop run; raise `beta` from 0.001 to 0.01 |
| `completion_length` truncates at 8192 every step | DAPO overlong shaping inactive | Verify `mask_truncated_completions: true` in YAML |
| Pod self-destroyed at $X | budget tripwire | Check Slack; raise `VRM_MAX_USD_GRPO` if intentional |
| HF upload 401 | token expired | Rotate `HF_TOKEN`, re-add via `gh secret set` |

## 10. Escalation

For RunPod incidents (pod stuck, GPU offline): https://www.runpod.io/console/help → priority ticket if Secure Cloud.
For HF Hub incidents: https://status.huggingface.co.
For W&B incidents: https://status.wandb.ai.
```

- [ ] **Step 2: Commit**

```bash
git add projects/VRM-model/docs/runbook.md
git commit -m "vrm: add operations runbook (commands, triage, escalation)"
```

---

### Task 2: `docs/budgets.md`

**Files:**
- Create: `projects/VRM-model/docs/budgets.md`

- [ ] **Step 1: Write**

`projects/VRM-model/docs/budgets.md`:

```markdown
# VRM-7B Budgets, Cost Rates, & Tripwires

## 1. RunPod Secure Cloud rates (verify quarterly)

| GPU type | $/hr per GPU | 8×GPU $/hr |
|---|---|---|
| NVIDIA H100 80GB SXM | $2.99 | $23.92 |
| NVIDIA H200 SXM 141GB (default) | $3.99 | $31.92 |
| NVIDIA B200 Blackwell | $5.99 | $47.92 |
| CPU pod (dataprep) | $0.04 | — |

| Storage | Cost |
|---|---|
| Network volume | $0.07 / GB / month |
| 2 TB volume (default) | $140 / month |

Source: RunPod console (Secure Cloud, US, on-demand). Edit `vrm/infra/budget.py` if rates change.

## 2. Per-stage budgets

| Stage | Wall time | Compute spend | Tripwire (USD) | Env var |
|---|---|---|---|---|
| Data build (CPU + teacher API) | 6-12 h | $50 + $3-5K teacher API | $500 | `VRM_MAX_USD_DATAPREP` |
| Stage 1 SFT (8×H200, 24 h) | ~24 h | ~$800 | $1500 | `VRM_MAX_USD_SFT` |
| Stage 2 GRPO (8×H200, 14 d) | 10-14 d | ~$8K | $8000 | `VRM_MAX_USD_GRPO` |
| Stage 3 rejection SFT (optional) | ~24 h | ~$800 | $1500 | `VRM_MAX_USD_REJECTION` |
| Eval — full (1×H200, 8 h) | ~8 h | ~$32 | $200 | `VRM_MAX_USD_EVAL` |
| Eval — quick (1×H200, 1 h) | ~1 h | ~$4 | $200 | `VRM_MAX_USD_EVAL` |

**Tripwire behavior:** the in-pod `vrm.infra.budget` daemon polls every 60s, computes `elapsed_h × hourly_burn_rate`, and:
1. POSTs `vrm-{stage}-failure` with `reason=budget_tripwire` to GH.
2. POSTs to Slack webhook if `SLACK_WEBHOOK_VRM` set.
3. Calls RunPod `DELETE /pods/{id}` to self-terminate.

The tripwire CANNOT be disabled via the pod's env; only by raising the limit.

## 3. Total project budget envelope

| Phase | Spend |
|---|---|
| Bootstrap + smoke runs | $200 |
| Data build (one good shard) | $4,500 (incl. teacher API) |
| Stage 1 SFT | $800 |
| Stage 2 GRPO | $8,000 |
| Stage 3 (optional) | $800 |
| Eval iterations (5 full evals) | $200 |
| Re-runs + slack | $2,500 |
| **Hard cap** | **$17,000** |

Add **$140/mo** for the network volume as long as it exists.

## 4. Cost mitigation playbook

1. **Use spot pods only for re-runnable jobs** — Stage 1 SFT (24h) is short enough; for GRPO use on-demand.
2. **Lower `num_generations` from 8 → 4** for early experiments — halves rollout cost; raise back for the final run.
3. **Use `quick` eval suite during dev**; full suite only on release-candidate checkpoints.
4. **Cache HF datasets on the network volume** — re-downloading a 200 GB shard costs $0 in transfer but ~$15 in idle pod time on every cold start.
5. **Skip Stage 3** unless eval shows headroom on MathVision/OlympiadBench — saves $1K.

## 5. When to escalate

| Spend overrun | Action |
|---|---|
| 1.1× tripwire | Routine — pod auto-killed; no action |
| 2× any single stage budget | Pause runs; review W&B for divergence; likely a config bug |
| Cumulative project >$15K with no shippable model | Stop; reconsider hyperparameters or fall back to LoRA Stage 1 |
| Teacher API >$5K and not yet usable | Pause distillation; switch to single-teacher (Claude only) for next batch |
```

- [ ] **Step 2: Commit**

```bash
git add projects/VRM-model/docs/budgets.md
git commit -m "vrm: add budgets + cost-tripwire policy doc"
```

---

### Task 3: Update the index plan with completion status

**Files:**
- Modify: `projects/VRM-model/docs/plan.md`

- [ ] **Step 1: As you complete each sub-plan, flip the status in the table from ☐ to ✅**

Example final state:
```markdown
| 00 | [bootstrap](plans/00-bootstrap.md) | ... | ✅ |
```

- [ ] **Step 2: Once all 14 are ✅, commit the final index update**

```bash
git add projects/VRM-model/docs/plan.md
git commit -m "vrm: mark all sub-plans complete in index"
```

---

## Done when

- [ ] Both `docs/runbook.md` and `docs/budgets.md` exist and are reviewed by Sumit.
- [ ] Index at `docs/plan.md` shows all 14 sub-plans ✅.
- [ ] Project is shippable: any teammate can read the runbook and operate the system end-to-end.
