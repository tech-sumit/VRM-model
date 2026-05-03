import json

import httpx
import respx

from vrm.infra.webhook import post_status


@respx.mock
def test_post_status_repository_dispatch(monkeypatch):
    monkeypatch.setenv("GH_TOKEN_FOR_DISPATCH", "ghp_xxx")
    monkeypatch.setenv("GH_REPO", "tech-sumit/react-webmcp")
    monkeypatch.delenv("SLACK_WEBHOOK_VRM", raising=False)
    route = respx.post("https://api.github.com/repos/tech-sumit/react-webmcp/dispatches").mock(
        return_value=httpx.Response(204)
    )

    post_status(
        "completed",
        task="sft",
        run_name="r1",
        payload={"checkpoint": "tech-sumit/x"},
    )
    assert route.called
    body = json.loads(route.calls[0].request.content)
    assert body["event_type"] == "vrm-sft-completed"
    assert body["client_payload"]["checkpoint"] == "tech-sumit/x"


@respx.mock
def test_post_status_silent_when_no_gh_token(monkeypatch):
    monkeypatch.delenv("GH_TOKEN_FOR_DISPATCH", raising=False)
    monkeypatch.delenv("GH_REPO", raising=False)
    monkeypatch.delenv("SLACK_WEBHOOK_VRM", raising=False)
    post_status("started", task="sft", run_name="r1")
    assert len(respx.calls) == 0
