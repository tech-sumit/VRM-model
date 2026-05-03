"""Post lifecycle events from inside a pod back to GitHub via repository_dispatch.

GH Actions can listen to `repository_dispatch` events to trigger downstream
workflows (e.g., kick off `vrm-eval.yml` once `vrm-train-grpo` completes).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Literal

import httpx

Status = Literal["started", "checkpoint", "completed", "failure"]


def _slack_payload(
    status: Status, task: str, run_name: str, payload: dict[str, Any] | None
) -> dict[str, Any]:
    color = {
        "started": "#888",
        "checkpoint": "#0bf",
        "completed": "#0c0",
        "failure": "#c00",
    }[status]
    text = f"*VRM {task}* `{run_name}` -- {status}"
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
        except Exception as e:
            print(f"[webhook] GH dispatch failed: {e}", file=sys.stderr)

    slack_url = os.environ.get("SLACK_WEBHOOK_VRM")
    if slack_url:
        try:
            httpx.post(
                slack_url,
                json=_slack_payload(status, task, run_name, payload),
                timeout=10.0,
            )
        except Exception as e:
            print(f"[webhook] Slack post failed: {e}", file=sys.stderr)


def main() -> None:
    """CLI: vrm-webhook <status> <task> <run_name> [<json-payload>]"""
    args = sys.argv[1:]
    if len(args) < 3:
        print(
            "usage: vrm-webhook <status> <task> <run_name> [<json-payload>]",
            file=sys.stderr,
        )
        sys.exit(2)
    status, task, run_name = args[:3]
    payload = json.loads(args[3]) if len(args) > 3 else None
    post_status(status, task=task, run_name=run_name, payload=payload)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
