"""Smoke tests for each registered normalizer.

Fixtures match the real HF schemas (verified 2026-05-03). Tests assert
each normalizer accepts a representative row, drops rows with missing
required fields, and produces a valid Record.
"""

import pytest

from vrm.data.normalize.chartqa import normalize as chartqa_normalize
from vrm.data.normalize.geo170k import normalize as geo170k_normalize
from vrm.data.normalize.geometry3k import normalize as geometry3k_normalize
from vrm.data.normalize.geoqa import normalize as geoqa_normalize
from vrm.data.normalize.mathv360k import normalize as mathv360k_normalize
from vrm.data.normalize.mavis import normalize as mavis_normalize
from vrm.data.normalize.vision_r1_cold import normalize as vision_r1_cold_normalize

_MAVIS_RAW = {
    "image_text_lite": "/tmp/x.png",
    "text_en": {"text_lite_question": "What is 2+2?"},
    "choices": ["3", "4", "5"],
    "answer_index": 1,
}
_CHARTQA_RAW = {
    "image": "/tmp/x.png",
    "query": "What is the value of bar A?",
    "label": ["42"],
}
_GEOMETRY3K_RAW = {
    "images": ["/tmp/x.png"],
    "problem": "<image>Find x.",
    "answer": "A",
}
_GEOQA_RAW = {
    "image": "/tmp/x.png",
    "problem": "Find angle 2.",
    "solution": "<answer> 145° </answer>",
}
_LLAVA_RAW = {
    "id": "row-0",
    "image": "coco/train2017/000000.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nSolve this."},
        {"from": "gpt", "value": "<think>...</think><answer>42</answer>"},
    ],
}


@pytest.mark.parametrize(
    ("normalize", "source", "raw"),
    [
        (mavis_normalize, "mavis", _MAVIS_RAW),
        (chartqa_normalize, "chartqa", _CHARTQA_RAW),
        (geometry3k_normalize, "geometry3k", _GEOMETRY3K_RAW),
        (geoqa_normalize, "geoqa", _GEOQA_RAW),
        (mathv360k_normalize, "mathv360k", _LLAVA_RAW),
        (vision_r1_cold_normalize, "vision_r1_cold", _LLAVA_RAW),
        (geo170k_normalize, "geo170k", _LLAVA_RAW),
    ],
)
def test_basic_normalize_returns_record(normalize, source, raw):
    rec = normalize(raw)
    assert rec is not None, f"{source} normalize returned None"
    assert rec.source == source
    assert rec.images, f"{source} record has no images"
    assert rec.answer
    assert rec.user_text()


def test_mavis_drops_when_required_fields_missing():
    assert mavis_normalize({**_MAVIS_RAW, "image_text_lite": None}) is None
    assert mavis_normalize({**_MAVIS_RAW, "choices": []}) is None
    assert mavis_normalize({**_MAVIS_RAW, "answer_index": None}) is None
    assert mavis_normalize({**_MAVIS_RAW, "answer_index": 99}) is None


def test_chartqa_drops_without_label():
    assert chartqa_normalize({**_CHARTQA_RAW, "label": []}) is None
    assert chartqa_normalize({**_CHARTQA_RAW, "image": None}) is None


def test_geometry3k_drops_without_answer():
    assert geometry3k_normalize({**_GEOMETRY3K_RAW, "answer": ""}) is None
    assert geometry3k_normalize({**_GEOMETRY3K_RAW, "images": []}) is None


def test_geoqa_extracts_from_answer_tags():
    rec = geoqa_normalize(_GEOQA_RAW)
    assert rec is not None
    assert rec.answer == "145"  # degree symbol stripped, numeric
    assert rec.answer_type == "numeric"


def test_llava_drops_without_conversations():
    assert mathv360k_normalize({**_LLAVA_RAW, "conversations": []}) is None
    assert mathv360k_normalize({**_LLAVA_RAW, "image": None}) is None
