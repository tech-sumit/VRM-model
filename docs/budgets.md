# VRM-7B Budgets

Cost rates and per-stage budgets for RunPod Secure Cloud.
Tripwires live in `src/vrm/infra/budget.py` and are read from env vars
`VRM_MAX_USD_*` set by the GH Actions workflows.

## RunPod Secure Cloud rates (US, on-demand, mid-2026)

> Verify against RunPod console before each major run. Rates change
> regularly; spot/community-cloud is 30-60% cheaper but evicts.

| GPU | Memory | Hourly | Notes |
|---|---|---|---|
| NVIDIA H100 80GB SXM | 80 GB | $2.99 | Older sibling of H200 |
| NVIDIA H200 SXM | 141 GB | $3.99 | **default for VRM** |
| NVIDIA B200 (Blackwell) | 192 GB | $5.99 | Future option (faster, pricier) |
| CPU | — | $0.04 | dataprep / orchestration |

Storage: $0.07/GB/month for network volumes (so a 2TB vol = $140/mo,
or about **$0.19/hr** flat).

## Per-stage budget envelope

Computed for an 8×H200 pod (= $31.92/hr GPU + $0.19/hr storage = **$32.11/hr**).
The per-stage `max_usd` includes a ~25% safety buffer on top of expected duration.

| Stage | Expected duration | Expected cost | Tripwire (`VRM_MAX_USD_*`) |
|---|---|---|---|
| dataprep | 8-24h (CPU + small-GPU side jobs + teacher API) | $300-500 | `$500` |
| Stage 1 SFT (full) | 18-30h on 8×H200 | $580-960 | `$1500` |
| Stage 1 SFT (LoRA) | 8-14h on 8×H200 | $260-450 | `$700` |
| Stage 2 GRPO | 10-14d on 8×H200 | $7,700-10,800 | `$8000` |
| Stage 3 rejection SFT (optional) | 4-8h on 8×H200 | $130-260 | `$500` |
| Eval (full suite) | 4-8h on 1×H200 | $16-32 | `$200` |
| **Total per release cycle** | ~12-16d | **~$8,800-12,500** | — |

Tripwires intentionally tight; raise via `vars` in GH Actions if hit.

## Teacher API costs (one-time per dataset version)

Per spec §3.3 we ensemble Claude Opus and GPT-5; rough budgets per 280k SFT
records (one solution per teacher, picked-best, ~2k input + ~3k output tokens):

| Model | Input rate | Output rate | Total cost (~280k records) |
|---|---|---|---|
| Claude Opus 4.7 thinking | $15/1M | $75/1M | ~$200-300 |
| GPT-5.4 medium | $10/1M | $30/1M | ~$80-120 |

Plus the `pass@K` filter inference cost (~$50 self-hosted on a small GPU).

## Cost mitigation

- Re-use base SFT data versions across GRPO experiments (most cost is in SFT data prep)
- Use Stage 1 LoRA (`mode=lora`) for ablations
- For exploration runs, set `--limit` on `vrm data normalize` to subsample
  before paying for teacher distillation
- For GRPO, lower `--max_steps` from 4000 to 1000 for a "v0" run before
  committing to a 14-day production run
- Cache HF datasets and model snapshots on the network volume (already configured)

## Monitoring

- Grafana dashboard `vrm` shows GPU util, throughput, eval delta
- Slack `#vrm-ops` channel: lifecycle events (started/checkpoint/completed/failure)
- W&B project `vrm-7b`: training curves, sample completions
- RunPod console: per-pod current spend (the source of truth)

## When to escalate

- Sustained GPU util < 60% during training → investigate pipeline bottleneck
- Cost projection at step N > 1.3× plan → re-evaluate
- Eval delta < 5pp on key benchmarks (MathVision, MathVerse) → don't release;
  inspect reward shaping and data quality
