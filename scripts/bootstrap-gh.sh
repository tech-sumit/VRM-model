#!/usr/bin/env bash
# Bootstrap GitHub repo secrets + variables for VRM-7B workflows.
# Reads values from env (or prompts interactively if --interactive).
# Idempotent: safe to re-run.
#
# Usage:
#   bash scripts/bootstrap-gh.sh             # read from current env, skip empty
#   bash scripts/bootstrap-gh.sh --interactive   # prompt for missing
#   bash scripts/bootstrap-gh.sh --dry-run   # print what would be set
set -euo pipefail

INTERACTIVE=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --interactive) INTERACTIVE=true ;;
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
    esac
done

if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: gh CLI required (https://cli.github.com/)" >&2
    exit 2
fi

if ! gh auth status >/dev/null 2>&1; then
    echo "ERROR: gh not authenticated. Run: gh auth login" >&2
    exit 2
fi

REPO="${GH_REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null)}"
if [[ -z "$REPO" ]]; then
    echo "ERROR: cannot determine repo (set GH_REPO=<owner/name> or cd into a gh repo)" >&2
    exit 2
fi
echo "Target repo: $REPO"

# --- SECRETS: opaque runtime tokens ---
SECRET_NAMES=(
    RUNPOD_API_KEY
    HF_TOKEN
    WANDB_API_KEY
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    SLACK_WEBHOOK_VRM
    GH_TOKEN_FOR_DISPATCH
)

# --- VARIABLES: non-secret config (visible in Actions logs) ---
# Pairs: <name> <default-or-empty>
declare -a VAR_PAIRS=(
    "HF_ORG=tech-sumit"
    "WANDB_PROJECT=vrm-7b"
    "WANDB_ENTITY=tech-sumit"
    "VRM_TRAIN_IMAGE=ghcr.io/tech-sumit/vrm-train:latest"
    "VRM_EVAL_IMAGE=ghcr.io/tech-sumit/vrm-eval:latest"
    "VRM_DATAPREP_IMAGE=ghcr.io/tech-sumit/vrm-dataprep:latest"
    "VRM_NETWORK_VOLUME_ID="
    "VRM_REGION=US-GA-2"
    "VRM_GPU_TYPE_TRAIN=NVIDIA H200"
    "VRM_GPU_COUNT_TRAIN=8"
    "VRM_GPU_TYPE_EVAL=NVIDIA H200"
    "VRM_GPU_COUNT_EVAL=1"
    "VRM_MAX_USD_DATAPREP=500"
    "VRM_MAX_USD_SFT=1500"
    "VRM_MAX_USD_GRPO=8000"
    "VRM_MAX_USD_EVAL=200"
)

prompt_if_missing() {
    local name="$1" kind="$2" default="${3:-}"
    local current="${!name:-}"
    if [[ -z "$current" && "$INTERACTIVE" == "true" ]]; then
        if [[ "$kind" == "secret" ]]; then
            read -r -s -p "  $name (hidden): " value
            echo
        else
            read -r -p "  $name${default:+ [$default]}: " value
        fi
        [[ -z "$value" ]] && value="$default"
        current="$value"
    elif [[ -z "$current" && -n "$default" ]]; then
        current="$default"
    fi
    printf '%s' "$current"
}

set_secret() {
    local name="$1"
    local value
    value="$(prompt_if_missing "$name" secret)"
    if [[ -z "$value" ]]; then
        echo "  [skip] $name (no value)"
        return 0
    fi
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [dry-run] gh secret set $name (${#value} chars)"
    else
        printf '%s' "$value" | gh secret set "$name" --repo "$REPO"
        echo "  [ok] secret $name set"
    fi
}

set_variable() {
    local name="$1" default="$2"
    local value
    value="$(prompt_if_missing "$name" variable "$default")"
    if [[ -z "$value" ]]; then
        echo "  [skip] $name (no value)"
        return 0
    fi
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [dry-run] gh variable set $name=$value"
    else
        gh variable set "$name" --repo "$REPO" --body "$value" >/dev/null
        echo "  [ok] variable $name=$value"
    fi
}

echo ""
echo "=== Secrets ==="
for s in "${SECRET_NAMES[@]}"; do
    set_secret "$s"
done

echo ""
echo "=== Variables ==="
for pair in "${VAR_PAIRS[@]}"; do
    name="${pair%%=*}"
    default="${pair#*=}"
    set_variable "$name" "$default"
done

echo ""
echo "Done. Verify with: gh secret list --repo $REPO && gh variable list --repo $REPO"
