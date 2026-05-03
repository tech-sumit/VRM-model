from vrm.infra.budget import gpu_hour_cost, hourly_burn_rate


def test_gpu_hour_cost_h200_default():
    assert gpu_hour_cost("NVIDIA H200", count=8) == 8 * 3.99


def test_gpu_hour_cost_h100_default():
    assert gpu_hour_cost("NVIDIA H100 80GB SXM", count=8) == 8 * 2.99


def test_gpu_hour_cost_unknown_uses_conservative_default():
    assert gpu_hour_cost("Magic Quantum GPU", count=8) == 8 * 5.0


def test_hourly_burn_includes_storage():
    rate = hourly_burn_rate(gpu_type="NVIDIA H200", gpu_count=8, volume_gb=2000)
    # 8*3.99 = 31.92 GPU + (2000 * 0.07/720h ~= 0.194) storage
    assert 31.9 < rate < 32.2
