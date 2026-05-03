import json

import httpx
import pytest
import respx

from vrm.infra.runpod import PodSpec, RunPodClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    return RunPodClient(api_key="test-key")


@respx.mock
def test_create_pod_posts_correct_payload(client):
    route = respx.post("https://rest.runpod.io/v1/pods").mock(
        return_value=httpx.Response(
            200, json={"id": "pod-abc123", "machineId": "m1", "desiredStatus": "RUNNING"}
        )
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
    assert route.called
    body = json.loads(route.calls[0].request.content)
    assert body["gpuTypeId"] == "NVIDIA H200"
    assert body["gpuCount"] == 8
    assert {"key": "VRM_TASK", "value": "sft"} in body["env"]


@respx.mock
def test_destroy_pod_calls_delete(client):
    route = respx.delete("https://rest.runpod.io/v1/pods/pod-abc").mock(
        return_value=httpx.Response(200, json={})
    )
    client.destroy_pod("pod-abc")
    assert route.called
    assert route.calls[0].request.method == "DELETE"
