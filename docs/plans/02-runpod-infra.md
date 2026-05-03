# Sub-plan 02 — RunPod infra clients & pod orchestration

> Index: [`../plan.md`](../plan.md) · Depends on: [00 bootstrap](00-bootstrap.md) · Status: ☐

**Goal:** A small `vrm.infra.*` package that wraps the RunPod REST API, HuggingFace Hub, GitHub repository_dispatch webhooks, and an in-pod budget tripwire — so that GH Actions / Makefile commands can launch and monitor pods with two-line invocations.

**Architecture:** Four narrowly-scoped modules. `runpod.py` for pod create/destroy. `hf_hub.py` for checkpoint upload helpers. `webhook.py` for posting back to GH on milestones. `budget.py` is a daemon that runs alongside training inside the pod, tracks GPU-hours via `nvidia-smi`, and self-shuts the pod when an env-configured USD limit is hit.

**Tech Stack:** httpx · tenacity (retries) · click · pydantic · huggingface-hub · pynvml.

---

### Task 1: RunPod REST client (`vrm/infra/runpod.py`)

**Files:**
- Create: `projects/VRM-model/src/vrm/infra/runpod.py`
- Create: `projects/VRM-model/tests/unit/test_runpod_client.py`

- [ ] **Step 1: Write the failing test**

`projects/VRM-model/tests/unit/test_runpod_client.py`:

```python
import pytest
import responses

from vrm.infra.runpod import RunPodClient, PodSpec


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    return RunPodClient(api_key="test-key")


@responses.activate
def test_create_pod_posts_correct_payload(client):
    responses.add(
        responses.POST,
        "https://rest.runpod.io/v1/pods",
        json={"id": "pod-abc123", "machineId": "m1", "desiredStatus": "RUNNING"},
        status=200,
    )
    spec = PodSpec(
        name="vrm-sft-test",
        image="ghcr.io/tech-sumit/vrm-train:latest",
        gpu_type_id="NVIDIA H200",
        gpu_count=8,
        volume_id="vol-xyz",
        volume_mount_path="/workspace/data",
        env={"VRM_TASK": "sft", "RUN_NAME": "test"},
        region="US-GA-2",
    )
    pod_id = client.create_pod(spec)
    assert pod_id == "pod-abc123"
    body = responses.calls[0].request.body
    assert b'"gpuTypeId":"NVIDIA H200"' in body
    assert b'"gpuCount":8' in body
    assert b'"VRM_TASK"' in body


@responses.activate
def test_destroy_pod_calls_delete(client):
    responses.add(responses.DELETE, "https://rest.runpod.io/v1/pods/pod-abc", status=200)
    client.destroy_pod("pod-abc")
    assert len(responses.calls) == 1
    assert responses.calls[0].request.method == "DELETE"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
make test
```
Expected: FAIL with `ModuleNotFoundError: No module named 'vrm.infra.runpod'`.

- [ ] **Step 3: Write the implementation**

`projects/VRM-model/src/vrm/infra/runpod.py`:

```python
"""RunPod REST API v1 client. Minimal: create_pod, get_pod, destroy_pod, list_pods, ssh_endpoint.

Docs: https://rest.runpod.io/v1/docs (verify endpoint shape against current API; this client targets v1).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

import click
import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

DEFAULT_BASE_URL = "https://rest.runpod.io/v1"


@dataclass(frozen=True)
class PodSpec:
    """Minimal spec for a RunPod pod we care about."""
    name: str
    image: str
    gpu_type_id: str  # e.g. "NVIDIA H200"
    gpu_count: int
    volume_id: str | None = None
    volume_mount_path: str = "/workspace/data"
    container_disk_in_gb: int = 200
    env: dict[str, str] = field(default_factory=dict)
    region: str | None = None
    cloud_type: str = "SECURE"  # "SECURE" or "COMMUNITY"
    ports: str = "22/tcp,8000/http"  # ssh + optional rollout server

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "imageName": self.image,
            "gpuTypeId": self.gpu_type_id,
            "gpuCount": self.gpu_count,
            "volumeInGb": 0,
            "containerDiskInGb": self.container_disk_in_gb,
            "networkVolumeId": self.volume_id,
            "volumeMountPath": self.volume_mount_path,
            "env": [{"key": k, "value": v} for k, v in self.env.items()],
            "ports": self.ports,
            "cloudType": self.cloud_type,
            **({"region": self.region} if self.region else {}),
        }


class RunPodError(RuntimeError):
    pass


class RunPodClient:
    def __init__(self, api_key: str | None = None, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise RunPodError("RUNPOD_API_KEY must be set (env or constructor arg).")
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    def __enter__(self) -> "RunPodClient":
        return self

    def __exit__(self, *_: object) -> None:
        self._client.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post(path, json=payload)
        if r.status_code >= 400:
            raise RunPodError(f"POST {path} {r.status_code}: {r.text}")
        return r.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _get(self, path: str) -> dict[str, Any]:
        r = self._client.get(path)
        if r.status_code >= 400:
            raise RunPodError(f"GET {path} {r.status_code}: {r.text}")
        return r.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _delete(self, path: str) -> None:
        r = self._client.delete(path)
        if r.status_code >= 400:
            raise RunPodError(f"DELETE {path} {r.status_code}: {r.text}")

    def create_pod(self, spec: PodSpec) -> str:
        data = self._post("/pods", spec.to_payload())
        pod_id = data.get("id")
        if not pod_id:
            raise RunPodError(f"create_pod missing 'id': {data}")
        return pod_id

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self._get(f"/pods/{pod_id}")

    def destroy_pod(self, pod_id: str) -> None:
        self._delete(f"/pods/{pod_id}")

    def list_pods(self) -> list[dict[str, Any]]:
        data = self._get("/pods")
        return data.get("pods", []) if isinstance(data, dict) else data


# --- CLI: `python -m vrm.infra.runpod ...` ---

@click.group()
def cli() -> None:
    """Launch and manage RunPod pods for VRM workloads."""


def _common_env() -> dict[str, str]:
    """Env vars passed into every pod from the launching shell."""
    keys = [
        "HF_TOKEN", "WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY",
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "SLACK_WEBHOOK_VRM",
        "GH_REPO", "GH_TOKEN_FOR_DISPATCH",
        "VRM_GIT_REPO", "VRM_GIT_REF",
    ]
    return {k: os.environ[k] for k in keys if os.environ.get(k)}


def _make_spec(name: str, image: str, gpu_type: str, gpu_count: int, env: dict[str, str]) -> PodSpec:
    return PodSpec(
        name=name,
        image=image,
        gpu_type_id=gpu_type,
        gpu_count=gpu_count,
        volume_id=os.environ.get("VRM_NETWORK_VOLUME_ID"),
        env=env,
        region=os.environ.get("VRM_REGION"),
    )


@cli.command("launch-train")
@click.option("--stage", type=click.Choice(["sft", "grpo", "rejection"]), required=True)
@click.option("--config", required=True, help="Path inside repo to the YAML config")
@click.option("--data-version", required=True)
@click.option("--run-name", required=True)
@click.option("--sft-checkpoint", default=None)
@click.option("--grpo-checkpoint", default=None)
def launch_train(stage: str, config: str, data_version: str, run_name: str,
                 sft_checkpoint: str | None, grpo_checkpoint: str | None) -> None:
    env = _common_env() | {
        "VRM_TASK": stage,
        "VRM_CONFIG": config,
        "DATA_VERSION": data_version,
        "RUN_NAME": run_name,
        "VRM_MAX_USD": os.environ.get(f"VRM_MAX_USD_{stage.upper()}", "5000"),
    }
    if sft_checkpoint:
        env["SFT_CHECKPOINT"] = sft_checkpoint
    if grpo_checkpoint:
        env["GRPO_CHECKPOINT"] = grpo_checkpoint
    spec = _make_spec(
        name=f"vrm-{stage}-{run_name}",
        image=os.environ.get("VRM_TRAIN_IMAGE", "ghcr.io/tech-sumit/vrm-train:latest"),
        gpu_type=os.environ.get("VRM_GPU_TYPE_TRAIN", "NVIDIA H200"),
        gpu_count=int(os.environ.get("VRM_GPU_COUNT_TRAIN", "8")),
        env=env,
    )
    with RunPodClient() as c:
        pod_id = c.create_pod(spec)
    click.echo(pod_id)


@cli.command("launch-eval")
@click.option("--checkpoint", required=True)
@click.option("--suite", default="full")
def launch_eval(checkpoint: str, suite: str) -> None:
    run_name = f"eval-{suite}-{checkpoint.split('/')[-1]}"
    env = _common_env() | {
        "VRM_TASK": "eval",
        "CHECKPOINT": checkpoint,
        "SUITE": suite,
        "RUN_NAME": run_name,
        "VRM_MAX_USD": os.environ.get("VRM_MAX_USD_EVAL", "200"),
    }
    spec = _make_spec(
        name=f"vrm-eval-{run_name}",
        image=os.environ.get("VRM_EVAL_IMAGE", "ghcr.io/tech-sumit/vrm-eval:latest"),
        gpu_type=os.environ.get("VRM_GPU_TYPE_EVAL", "NVIDIA H200"),
        gpu_count=int(os.environ.get("VRM_GPU_COUNT_EVAL", "1")),
        env=env,
    )
    with RunPodClient() as c:
        pod_id = c.create_pod(spec)
    click.echo(pod_id)


@cli.command("launch-dataprep")
@click.option("--recipe", "recipes", multiple=True, required=True)
@click.option("--data-version", required=True)
def launch_dataprep(recipes: tuple[str, ...], data_version: str) -> None:
    env = _common_env() | {
        "VRM_TASK": "dataprep",
        "VRM_CONFIG": ",".join(recipes),
        "DATA_VERSION": data_version,
        "RUN_NAME": f"dataprep-{data_version}",
        "VRM_MAX_USD": os.environ.get("VRM_MAX_USD_DATAPREP", "500"),
    }
    spec = _make_spec(
        name=f"vrm-dataprep-{data_version}",
        image=os.environ.get("VRM_DATAPREP_IMAGE", "ghcr.io/tech-sumit/vrm-dataprep:latest"),
        gpu_type="CPU",  # CPU pods on RunPod
        gpu_count=0,
        env=env,
    )
    with RunPodClient() as c:
        pod_id = c.create_pod(spec)
    click.echo(pod_id)


@cli.command("destroy")
@click.argument("pod_id")
def destroy(pod_id: str) -> None:
    with RunPodClient() as c:
        c.destroy_pod(pod_id)
    click.echo(f"destroyed {pod_id}")


@cli.command("status")
@click.argument("pod_id")
def status(pod_id: str) -> None:
    with RunPodClient() as c:
        click.echo(c.get_pod(pod_id))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
make test
```
Expected: PASS · 2 passed.

- [ ] **Step 5: Commit**

```bash
git add projects/VRM-model/src/vrm/infra/runpod.py projects/VRM-model/tests/unit/test_runpod_client.py
git commit -m "vrm: add RunPod REST client with pod create/destroy/launch CLI"
```

---

### Task 2: GH webhook poster (`vrm/infra/webhook.py`)

**Files:**
- Create: `projects/VRM-model/src/vrm/infra/webhook.py`
- Create: `projects/VRM-model/tests/unit/test_webhook.py`

- [ ] **Step 1: Write the failing test**

```python
# projects/VRM-model/tests/unit/test_webhook.py
import pytest
import responses

from vrm.infra.webhook import post_status


@responses.activate
def test_post_status_repository_dispatch(monkeypatch):
    monkeypatch.setenv("GH_TOKEN_FOR_DISPATCH", "ghp_xxx")
    monkeypatch.setenv("GH_REPO", "tech-sumit/react-webmcp")
    responses.add(
        responses.POST,
        "https://api.github.com/repos/tech-sumit/react-webmcp/dispatches",
        status=204,
    )
    post_status("completed", task="sft", run_name="r1", payload={"checkpoint": "tech-sumit/x"})
    assert len(responses.calls) == 1
    body = responses.calls[0].request.body
    assert b"vrm-sft-completed" in body
    assert b"tech-sumit/x" in body
```

- [ ] **Step 2: Run test (FAILS)**

```bash
make test
```

- [ ] **Step 3: Write implementation**

`projects/VRM-model/src/vrm/infra/webhook.py`:

```python
"""Post lifecycle events from inside a pod back to GitHub via repository_dispatch.

GH Actions can listen to `repository_dispatch` events to trigger downstream workflows
(e.g., kick off `vrm-eval.yml` once `vrm-train-grpo` completes).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Literal

import httpx

Status = Literal["started", "checkpoint", "completed", "failure"]


def _slack_payload(status: Status, task: str, run_name: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    color = {"started": "#888", "checkpoint": "#0bf", "completed": "#0c0", "failure": "#c00"}[status]
    text = f"*VRM {task}* `{run_name}` — {status}"
    if payload:
        text += "\n```" + json.dumps(payload, indent=2)[:1500] + "```"
    return {"attachments": [{"color": color, "text": text}]}


def post_status(
    status: Status,
    *,
    task: str,
    run_name: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Post a status event to GitHub repository_dispatch and optionally Slack.

    Silent no-op if GH_TOKEN_FOR_DISPATCH is unset (allows local dry-runs).
    """
    payload = payload or {}
    payload = {"task": task, "run_name": run_name, **payload}

    gh_token = os.environ.get("GH_TOKEN_FOR_DISPATCH")
    gh_repo = os.environ.get("GH_REPO")
    if gh_token and gh_repo:
        event_type = f"vrm-{task}-{status}"
        try:
            r = httpx.post(
                f"https://api.github.com/repos/{gh_repo}/dispatches",
                headers={
                    "Authorization": f"Bearer {gh_token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                json={"event_type": event_type, "client_payload": payload},
                timeout=10.0,
            )
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            print(f"[webhook] GH dispatch failed: {e}", file=sys.stderr)

    slack_url = os.environ.get("SLACK_WEBHOOK_VRM")
    if slack_url:
        try:
            httpx.post(slack_url, json=_slack_payload(status, task, run_name, payload), timeout=10.0)
        except Exception as e:  # noqa: BLE001
            print(f"[webhook] Slack post failed: {e}", file=sys.stderr)


def main() -> None:
    """CLI: vrm-webhook <status> <task> <run_name> [<json-payload>]"""
    args = sys.argv[1:]
    if len(args) < 3:
        print("usage: vrm-webhook <status> <task> <run_name> [<json-payload>]", file=sys.stderr)
        sys.exit(2)
    status, task, run_name = args[:3]
    payload = json.loads(args[3]) if len(args) > 3 else None
    post_status(status, task=task, run_name=run_name, payload=payload)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test (PASSES)**

- [ ] **Step 5: Commit**

```bash
git add projects/VRM-model/src/vrm/infra/webhook.py projects/VRM-model/tests/unit/test_webhook.py
git commit -m "vrm: add GH repository_dispatch + Slack webhook poster"
```

---

### Task 3: Budget tripwire daemon (`vrm/infra/budget.py`)

**Files:**
- Create: `projects/VRM-model/src/vrm/infra/budget.py`
- Create: `projects/VRM-model/tests/unit/test_budget.py`

- [ ] **Step 1: Write the failing test (focus on cost-rate calculation, mock GPU detection)**

```python
# projects/VRM-model/tests/unit/test_budget.py
from vrm.infra.budget import gpu_hour_cost, hourly_burn_rate


def test_gpu_hour_cost_h200_default():
    assert gpu_hour_cost("NVIDIA H200", count=8) == 8 * 3.99


def test_gpu_hour_cost_h100_default():
    assert gpu_hour_cost("NVIDIA H100 80GB SXM", count=8) == 8 * 2.99


def test_hourly_burn_includes_storage():
    rate = hourly_burn_rate(gpu_type="NVIDIA H200", gpu_count=8, volume_gb=2000)
    # 8*3.99 = 31.92 GPU + (2000 * 0.07/720h ≈ 0.194) storage
    assert 31.9 < rate < 32.2
```

- [ ] **Step 2: Run test (FAILS)**

- [ ] **Step 3: Write implementation**

`projects/VRM-model/src/vrm/infra/budget.py`:

```python
"""Budget tripwire daemon.

Runs as a background process inside the pod (see scripts/pod-entrypoint.sh).
Polls elapsed time, computes cumulative spend at the GPU+storage hourly rate,
and self-destroys the pod via the RunPod API when the limit is exceeded.

Reference rates (RunPod Secure Cloud, US, on-demand, mid-2026; verify against console):
- H100 SXM 80GB: $2.99/hr
- H200 SXM 141GB: $3.99/hr
- B200 (Blackwell): $5.99/hr
- Network volume storage: $0.07/GB/month
"""
from __future__ import annotations

import argparse
import os
import time

from vrm.infra.webhook import post_status

# These are wall-clock per-GPU rates on RunPod Secure Cloud, US region (verify).
GPU_HOUR_COST_USD: dict[str, float] = {
    "NVIDIA H100 80GB SXM": 2.99,
    "NVIDIA H200": 3.99,
    "NVIDIA B200": 5.99,
    "CPU": 0.04,  # generic CPU pod
}


def gpu_hour_cost(gpu_type: str, count: int) -> float:
    rate = GPU_HOUR_COST_USD.get(gpu_type)
    if rate is None:
        return 5.0 * count  # conservative default
    return rate * count


def hourly_burn_rate(gpu_type: str, gpu_count: int, volume_gb: int) -> float:
    """USD/hr for compute + storage."""
    storage_per_hour = (volume_gb * 0.07) / (24 * 30)  # $0.07/GB/month → /hr
    return gpu_hour_cost(gpu_type, gpu_count) + storage_per_hour


def _self_destruct(pod_id: str | None, task: str, run_name: str, spent: float, limit: float) -> None:
    payload = {"spent_usd": round(spent, 2), "limit_usd": limit, "pod_id": pod_id}
    post_status("failure", task=task, run_name=run_name, payload={"reason": "budget_tripwire", **payload})
    if pod_id and os.environ.get("RUNPOD_API_KEY"):
        # Late import to avoid pulling httpx in the daemon's hot path.
        from vrm.infra.runpod import RunPodClient
        try:
            with RunPodClient() as c:
                c.destroy_pod(pod_id)
        except Exception as e:  # noqa: BLE001
            print(f"[budget] failed to self-destroy: {e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--max-usd", type=float, required=True)
    p.add_argument("--poll-seconds", type=int, default=60)
    args = p.parse_args()

    pod_id = os.environ.get("RUNPOD_POD_ID")
    run_name = os.environ.get("RUN_NAME", "?")
    gpu_type = os.environ.get("VRM_GPU_TYPE_TRAIN", "NVIDIA H200")
    gpu_count = int(os.environ.get("VRM_GPU_COUNT_TRAIN", "8"))
    volume_gb = int(os.environ.get("VRM_VOLUME_GB", "2000"))

    burn = hourly_burn_rate(gpu_type, gpu_count, volume_gb)
    print(f"[budget] task={args.task} burn=${burn:.2f}/hr limit=${args.max_usd}")

    started = time.time()
    while True:
        elapsed_h = (time.time() - started) / 3600.0
        spent = elapsed_h * burn
        if spent >= args.max_usd:
            print(f"[budget] TRIPWIRE: spent=${spent:.2f} >= limit=${args.max_usd}")
            _self_destruct(pod_id, args.task, run_name, spent, args.max_usd)
            return
        if int(elapsed_h * 60) % 60 == 0:  # log roughly hourly
            print(f"[budget] elapsed={elapsed_h:.2f}h spent=${spent:.2f}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test (PASSES)**

- [ ] **Step 5: Commit**

```bash
git add projects/VRM-model/src/vrm/infra/budget.py projects/VRM-model/tests/unit/test_budget.py
git commit -m "vrm: add budget tripwire daemon (self-destroys pod on USD limit)"
```

---

### Task 4: HF Hub helpers (`vrm/infra/hf_hub.py`)

**Files:**
- Create: `projects/VRM-model/src/vrm/infra/hf_hub.py`
- Create: `projects/VRM-model/tests/unit/test_hf_hub.py`

- [ ] **Step 1: Write the failing test (focus on repo-id formatting; HfApi is mocked)**

```python
# projects/VRM-model/tests/unit/test_hf_hub.py
from vrm.infra.hf_hub import dataset_repo_id, model_repo_id


def test_dataset_repo_id_format():
    assert dataset_repo_id("sft", "v3") == "tech-sumit/vrm-7b-sft-v3"


def test_model_repo_id_format():
    assert model_repo_id("grpo", "2026-05-04") == "tech-sumit/vrm-7b-grpo-2026-05-04"
```

- [ ] **Step 2: Run (FAILS)**

- [ ] **Step 3: Write implementation**

`projects/VRM-model/src/vrm/infra/hf_hub.py`:

```python
"""Minimal HuggingFace Hub helpers — repo-id naming + checkpoint upload."""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

DEFAULT_ORG = os.environ.get("HF_ORG", "tech-sumit")


def dataset_repo_id(stage: str, data_version: str, org: str = DEFAULT_ORG) -> str:
    """e.g. dataset_repo_id('sft', 'v3') -> 'tech-sumit/vrm-7b-sft-v3'"""
    return f"{org}/vrm-7b-{stage}-{data_version}"


def model_repo_id(stage: str, run_name: str, org: str = DEFAULT_ORG) -> str:
    return f"{org}/vrm-7b-{stage}-{run_name}"


def upload_checkpoint(local_dir: Path, repo_id: str, token: str | None = None, *, private: bool = True) -> str:
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True, token=api.token)
    commit = api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, repo_type="model")
    return commit.commit_url if hasattr(commit, "commit_url") else str(commit)


def upload_dataset_shards(local_dir: Path, repo_id: str, token: str | None = None) -> str:
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True, token=api.token)
    commit = api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, repo_type="dataset")
    return commit.commit_url if hasattr(commit, "commit_url") else str(commit)
```

- [ ] **Step 4: Run (PASSES)**

- [ ] **Step 5: Commit**

```bash
git add projects/VRM-model/src/vrm/infra/hf_hub.py projects/VRM-model/tests/unit/test_hf_hub.py
git commit -m "vrm: add HF Hub repo-id helpers and checkpoint/dataset upload utilities"
```

---

### Task 5: Wire CLI subcommand into top-level `vrm` CLI

**Files:**
- Modify: `projects/VRM-model/src/vrm/cli.py`

- [ ] **Step 1: Add infra group to main CLI**

Replace the body of `src/vrm/cli.py` with:

```python
"""VRM CLI entrypoint."""
from __future__ import annotations

import click

from vrm import __version__
from vrm.infra.runpod import cli as runpod_cli


@click.group(help="VRM-7B training & evaluation toolchain")
@click.version_option(__version__)
def main() -> None:
    """Top-level CLI group."""


@main.command(help="Print package version and exit.")
def version() -> None:
    click.echo(__version__)


main.add_command(runpod_cli, name="runpod")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify**

```bash
uv run vrm runpod --help
```

Expected: shows `launch-train`, `launch-eval`, `launch-dataprep`, `destroy`, `status`.

- [ ] **Step 3: Commit**

```bash
git add projects/VRM-model/src/vrm/cli.py
git commit -m "vrm: expose runpod subcommands under top-level vrm CLI"
```

---

## Done when

- [ ] `make test` passes (now includes 7+ unit tests).
- [ ] `uv run vrm runpod launch-train --help` prints help.
- [ ] `python -m vrm.infra.budget --task sft --max-usd 1500 --poll-seconds 1` runs without error (will tripwire fast on a 0$ limit).
- [ ] `python -m vrm.infra.webhook started sft test-run` exits 0 (silent if no GH_TOKEN_FOR_DISPATCH).
