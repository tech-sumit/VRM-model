#!/usr/bin/env bash
# Pod entrypoint. Dispatches to a vrm.* module based on VRM_TASK env var.
# This script is the SINGLE entrypoint baked into all three images. The
# RunPod pod is started with VRM_TASK=<sft|grpo|rejection|eval|dataprep>
# and per-task env vars (DATA_VERSION, RUN_NAME, CHECKPOINT, ...).
set -euo pipefail

log() { echo "[$(date -Iseconds)] $*"; }
trap 'log "FATAL: line $LINENO failed"; python -m vrm.infra.webhook failure "${VRM_TASK:-?}" "${RUN_NAME:-?}" "{\"reason\":\"trap line $LINENO\"}" || true; exit 1' ERR

: "${VRM_TASK:?VRM_TASK env var is required (sft|grpo|rejection|eval|dataprep)}"
: "${RUN_NAME:?RUN_NAME env var is required}"

# Pull latest source on every cold start so we always run committed code.
if [[ -n "${VRM_GIT_REPO:-}" ]] && [[ -n "${VRM_GIT_REF:-}" ]]; then
    log "Pulling vrm source from $VRM_GIT_REPO@$VRM_GIT_REF"
    cd /workspace
    rm -rf vrm-src
    git clone "$VRM_GIT_REPO" vrm-src
    cd vrm-src
    git checkout "$VRM_GIT_REF"
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
        # VRM_CONFIG may be a comma-separated list of recipe YAML paths.
        IFS=',' read -ra _RECIPES <<< "${VRM_CONFIG:?}"
        _RECIPE_ARGS=()
        for r in "${_RECIPES[@]}"; do _RECIPE_ARGS+=("--recipe" "$r"); done
        _DISTILL_FLAG="--include-distillation"
        if [[ "${VRM_INCLUDE_DISTILLATION:-true}" == "false" ]]; then
            _DISTILL_FLAG="--no-distillation"
        fi
        exec python -m vrm.data.build \
            "${_RECIPE_ARGS[@]}" \
            --data-version "${DATA_VERSION:?}" \
            "$_DISTILL_FLAG"
        ;;
    *)
        log "Unknown VRM_TASK=$VRM_TASK"
        exit 2
        ;;
esac
