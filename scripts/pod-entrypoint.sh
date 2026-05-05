#!/usr/bin/env bash
# Pod entrypoint. Dispatches to a vrm.* module based on VRM_TASK env var.
# This script is the SINGLE entrypoint baked into all three images. The
# RunPod pod is started with VRM_TASK=<sft|grpo|rejection|eval|dataprep>
# and per-task env vars (DATA_VERSION, RUN_NAME, CHECKPOINT, ...).
#
# Debug/hotfix: set VRM_DEBUG_HOLD=1 in the pod env to make the pod SSH-
# accessible after the task exits (success or failure). Otherwise the pod
# terminates normally so RunPod billing stops.
set -uo pipefail   # NOTE: no -e so we can capture task exit code + hold the pod

log() { echo "[$(date -Iseconds)] $*"; }

hold_pod_if_debug() {
    local exit_code="$1"
    if [[ "${VRM_DEBUG_HOLD:-0}" == "1" ]]; then
        log "VRM_DEBUG_HOLD=1 -- sleeping forever (exit was $exit_code). SSH in to debug."
        # `wait` returns as soon as ANY child (sshd, budget daemon, or any
        # setsid-launched hotfix process) exits, which terminates the
        # container and lets RunPod reclaim the pod. Use sleep loop instead
        # so the entrypoint script stays blocked regardless of child state.
        while true; do sleep 3600; done
    fi
    exit "$exit_code"
}

: "${VRM_TASK:?VRM_TASK env var is required (sft|grpo|rejection|eval|dataprep)}"
: "${RUN_NAME:?RUN_NAME env var is required}"

# vLLM V1 worker subprocess start method. Default `fork` inherits all parent
# state (open R2 connections, asyncio loops, transformers caches) and crashes
# silently on Qwen2.5-VL multimodal profile. `spawn` starts a clean
# interpreter for the worker. Must be exported BEFORE any python invocation
# that imports vllm (transitively or directly).
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# Start sshd in the background for hotfix/debug access if RunPod injected a
# PUBLIC_KEY. Only starts when ssh is installed (all vrm-* images ship
# openssh-server). Non-fatal if keygen or sshd is missing.
if [[ -n "${PUBLIC_KEY:-}" ]] && command -v sshd >/dev/null 2>&1; then
    log "Starting sshd for hotfix access"
    mkdir -p /root/.ssh /run/sshd
    printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
    # Generate host keys if missing (fresh container).
    [[ ! -f /etc/ssh/ssh_host_ed25519_key ]] && ssh-keygen -A >/dev/null 2>&1 || true
    /usr/sbin/sshd -D -e &
    log "sshd listening on :22"
fi

# Pull latest source on every cold start so we always run committed code.
if [[ -n "${VRM_GIT_REPO:-}" ]] && [[ -n "${VRM_GIT_REF:-}" ]]; then
    log "Pulling vrm source from $VRM_GIT_REPO@$VRM_GIT_REF"
    cd /workspace
    rm -rf vrm-src
    git clone "$VRM_GIT_REPO" vrm-src
    cd vrm-src
    git checkout "$VRM_GIT_REF"
    # Use uv pip (bound to python3.11 on both train + dataprep images) instead of
    # distro `pip` which on Ubuntu 22.04 is bound to python3.10 and rejects our
    # requires-python >=3.11 constraint.
    if command -v uv >/dev/null 2>&1; then
        uv pip install --system --no-deps -e . || log "WARN: uv pip install failed, continuing with baked-in code"
    else
        python -m pip install --no-deps -e . || log "WARN: pip install failed, continuing with baked-in code"
    fi
fi

# Budget tripwire daemon (background)
python -m vrm.infra.budget --task "$VRM_TASK" --max-usd "${VRM_MAX_USD:?}" &
BUDGET_PID=$!
trap 'kill $BUDGET_PID 2>/dev/null || true' EXIT

log "Pod entrypoint: VRM_TASK=$VRM_TASK RUN_NAME=$RUN_NAME"
python -m vrm.infra.webhook started "$VRM_TASK" "$RUN_NAME"

task_rc=0
case "$VRM_TASK" in
    debug)
        # Persistent dev pod: skip all workload, just idle with sshd up so we
        # can iterate on code + run modules interactively without paying for
        # fresh CI builds + pod provisioning on every change.
        log "VRM_TASK=debug -- idling forever for interactive SSH use"
        while true; do sleep 3600; done
        ;;
    sft|rejection)
        python -m vrm.train.stage1_sft \
            --config "${VRM_CONFIG:?}" \
            --data-version "${DATA_VERSION:?}" \
            --run-name "$RUN_NAME" 2>&1 | tee /tmp/vrm-task.log
        task_rc=${PIPESTATUS[0]}
        ;;
    grpo)
        python -m vrm.train.stage2_grpo \
            --config "${VRM_CONFIG:?}" \
            --sft-checkpoint "${SFT_CHECKPOINT:?}" \
            --data-version "${DATA_VERSION:?}" \
            --run-name "$RUN_NAME" 2>&1 | tee /tmp/vrm-task.log
        task_rc=${PIPESTATUS[0]}
        ;;
    eval)
        python -m vrm.eval.run_vlmevalkit \
            --checkpoint "${CHECKPOINT:?}" \
            --suite "${SUITE:?}" \
            --run-name "$RUN_NAME" 2>&1 | tee /tmp/vrm-task.log
        task_rc=${PIPESTATUS[0]}
        ;;
    dataprep)
        # VRM_CONFIG may be a comma-separated list of recipe YAML paths.
        # VRM_STAGE selects normalize|filter|distill|all (default all for
        # backwards compatibility with single-pod runs).
        IFS=',' read -ra _RECIPES <<< "${VRM_CONFIG:?}"
        _RECIPE_ARGS=()
        for r in "${_RECIPES[@]}"; do _RECIPE_ARGS+=("--recipe" "$r"); done
        _DISTILL_FLAG="--include-distillation"
        if [[ "${VRM_INCLUDE_DISTILLATION:-true}" == "false" ]]; then
            _DISTILL_FLAG="--no-distillation"
        fi
        _STAGE="${VRM_STAGE:-all}"
        # Normalize + filter stages should NOT upload to HF Hub -- only the
        # final distill stage produces publish-ready shards.
        _UPLOAD_FLAG="--no-upload"
        if [[ "$_STAGE" == "distill" || "$_STAGE" == "all" ]]; then
            _UPLOAD_FLAG="--upload"
        fi
        python -m vrm.data.build \
            "${_RECIPE_ARGS[@]}" \
            --data-version "${DATA_VERSION:?}" \
            --stage "$_STAGE" \
            "$_DISTILL_FLAG" \
            "$_UPLOAD_FLAG" 2>&1 | tee /tmp/vrm-task.log
        task_rc=${PIPESTATUS[0]}
        ;;
    *)
        log "Unknown VRM_TASK=$VRM_TASK"
        hold_pod_if_debug 2
        ;;
esac

if [[ $task_rc -ne 0 ]]; then
    log "task exited with code $task_rc"
    tail -c 2000 /tmp/vrm-task.log 2>/dev/null > /tmp/vrm-tail.log || true
    python -m vrm.infra.webhook failure "$VRM_TASK" "$RUN_NAME" \
        "{\"exit_code\":$task_rc,\"tail\":$(python -c 'import json,sys; print(json.dumps(open("/tmp/vrm-tail.log").read() if __import__("os").path.exists("/tmp/vrm-tail.log") else ""))')}" || true
else
    log "task completed successfully"
    python -m vrm.infra.webhook completed "$VRM_TASK" "$RUN_NAME" || true
fi

hold_pod_if_debug "$task_rc"
