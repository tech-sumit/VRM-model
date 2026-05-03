"""Budget tripwire daemon.

Runs as a background process inside the pod (see scripts/pod-entrypoint.sh).
Polls elapsed time, computes cumulative spend at the GPU+storage hourly rate,
and self-destroys the pod via the RunPod API when the limit is exceeded.

Reference rates (RunPod Secure Cloud, US, on-demand, mid-2026; verify against
the console):
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

GPU_HOUR_COST_USD: dict[str, float] = {
    "NVIDIA H100 80GB SXM": 2.99,
    "NVIDIA H200": 3.99,
    "NVIDIA B200": 5.99,
    "CPU": 0.04,
}


def gpu_hour_cost(gpu_type: str, count: int) -> float:
    rate = GPU_HOUR_COST_USD.get(gpu_type)
    if rate is None:
        return 5.0 * count
    return rate * count


def hourly_burn_rate(gpu_type: str, gpu_count: int, volume_gb: int) -> float:
    """USD/hr for compute + storage."""
    storage_per_hour = (volume_gb * 0.07) / (24 * 30)
    return gpu_hour_cost(gpu_type, gpu_count) + storage_per_hour


def _self_destruct(pod_id: str | None, task: str, run_name: str, spent: float, limit: float) -> None:
    payload = {
        "spent_usd": round(spent, 2),
        "limit_usd": limit,
        "pod_id": pod_id,
    }
    post_status(
        "failure",
        task=task,
        run_name=run_name,
        payload={"reason": "budget_tripwire", **payload},
    )
    if pod_id and os.environ.get("RUNPOD_API_KEY"):
        from vrm.infra.runpod import RunPodClient

        try:
            with RunPodClient() as c:
                c.destroy_pod(pod_id)
        except Exception as e:
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
    print(
        f"[budget] task={args.task} burn=${burn:.2f}/hr limit=${args.max_usd}",
        flush=True,
    )

    started = time.time()
    while True:
        elapsed_h = (time.time() - started) / 3600.0
        spent = elapsed_h * burn
        if spent >= args.max_usd:
            print(
                f"[budget] TRIPWIRE: spent=${spent:.2f} >= limit=${args.max_usd}",
                flush=True,
            )
            _self_destruct(pod_id, args.task, run_name, spent, args.max_usd)
            return
        if int(elapsed_h * 60) % 60 == 0:
            print(f"[budget] elapsed={elapsed_h:.2f}h spent=${spent:.2f}", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
