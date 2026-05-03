"""RunPod REST API v1 client. Minimal: create_pod, get_pod, destroy_pod, list_pods.

Docs: https://rest.runpod.io/v1/docs (verify endpoint shape against current API; this
client targets v1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    gpu_type_id: str
    gpu_count: int
    volume_id: str | None = None
    volume_mount_path: str = "/workspace/data"
    container_disk_in_gb: int = 200
    env: dict[str, str] = field(default_factory=dict)
    region: str | None = None
    cloud_type: str = "SECURE"
    ports: str = "22/tcp,8000/http"

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        }
        if self.region:
            payload["region"] = self.region
        return payload


class RunPodError(RuntimeError):
    pass


class RunPodClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise RunPodError("RUNPOD_API_KEY must be set (env or constructor arg).")
        self._client = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def __enter__(self) -> RunPodClient:
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
        if isinstance(data, dict):
            return data.get("pods", [])
        return data


# --- CLI: `python -m vrm.infra.runpod ...` ---


@click.group()
def cli() -> None:
    """Launch and manage RunPod pods for VRM workloads."""


def _common_env() -> dict[str, str]:
    keys = [
        "HF_TOKEN",
        "WANDB_API_KEY",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "SLACK_WEBHOOK_VRM",
        "GH_REPO",
        "GH_TOKEN_FOR_DISPATCH",
        "VRM_GIT_REPO",
        "VRM_GIT_REF",
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
def launch_train(
    stage: str,
    config: str,
    data_version: str,
    run_name: str,
    sft_checkpoint: str | None,
    grpo_checkpoint: str | None,
) -> None:
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
@click.option(
    "--include-distillation/--no-distillation",
    default=True,
    show_default=True,
)
def launch_dataprep(recipes: tuple[str, ...], data_version: str, include_distillation: bool) -> None:
    env = _common_env() | {
        "VRM_TASK": "dataprep",
        "VRM_CONFIG": ",".join(recipes),
        "DATA_VERSION": data_version,
        "RUN_NAME": f"dataprep-{data_version}",
        "VRM_MAX_USD": os.environ.get("VRM_MAX_USD_DATAPREP", "500"),
        "VRM_INCLUDE_DISTILLATION": "true" if include_distillation else "false",
    }
    spec = _make_spec(
        name=f"vrm-dataprep-{data_version}",
        image=os.environ.get("VRM_DATAPREP_IMAGE", "ghcr.io/tech-sumit/vrm-dataprep:latest"),
        gpu_type="CPU",
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
