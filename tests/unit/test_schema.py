import pytest
from pydantic import ValidationError

from vrm.data.schema import Message, Record


def _valid_record() -> dict:
    return {
        "id": "mavis_00042",
        "images": ["s3://bucket/img.png"],
        "messages": [
            {"role": "system", "content": "Solve step-by-step."},
            {"role": "user", "content": "<image>\nIn triangle ABC..."},
            {
                "role": "assistant",
                "content": "<think>...</think><answer>72</answer>",
            },
        ],
        "answer": "72",
        "answer_type": "numeric",
        "verifier": "exact_numeric",
        "tolerance": 0.001,
        "difficulty": 0.42,
        "source": "mavis",
    }


def test_valid_record_parses():
    rec = Record.model_validate(_valid_record())
    assert rec.id == "mavis_00042"
    assert rec.answer_type == "numeric"
    assert rec.messages[-1].role == "assistant"


def test_record_rejects_unknown_answer_type():
    bad = _valid_record() | {"answer_type": "haiku"}
    with pytest.raises(ValidationError):
        Record.model_validate(bad)


def test_record_difficulty_bounded():
    bad = _valid_record() | {"difficulty": 1.5}
    with pytest.raises(ValidationError):
        Record.model_validate(bad)


def test_record_to_jsonl_roundtrip(tmp_path):
    rec = Record.model_validate(_valid_record())
    path = tmp_path / "out.jsonl"
    path.write_text(rec.model_dump_json() + "\n")
    line = path.read_text().splitlines()[0]
    rec2 = Record.model_validate_json(line)
    assert rec2 == rec


def test_message_role_enum_strict():
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "wizard", "content": "abracadabra"})
