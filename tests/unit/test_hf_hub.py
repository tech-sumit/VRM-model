from vrm.infra.hf_hub import dataset_repo_id, model_repo_id


def test_dataset_repo_id_format():
    assert dataset_repo_id("sft", "v3") == "tech-sumit/vrm-7b-sft-v3"


def test_dataset_repo_id_custom_org():
    assert dataset_repo_id("rl", "v2", org="other") == "other/vrm-7b-rl-v2"


def test_model_repo_id_format():
    assert model_repo_id("grpo", "2026-05-04") == "tech-sumit/vrm-7b-grpo-2026-05-04"
