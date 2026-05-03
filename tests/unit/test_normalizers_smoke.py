"""Smoke tests for each normalizer.

Each test uses a synthetic record matching the assumed schema. When the actual
HF dataset cards are verified, real fixtures should be added for the canonical
fields per source.
"""

import pytest

from vrm.data.normalize.chartqa import normalize as chartqa_normalize
from vrm.data.normalize.geo170k import normalize as geo170k_normalize
from vrm.data.normalize.geometry3k import normalize as geometry3k_normalize
from vrm.data.normalize.geoqa import normalize as geoqa_normalize
from vrm.data.normalize.mathv360k import normalize as mathv360k_normalize
from vrm.data.normalize.mathvista import normalize as mathvista_normalize
from vrm.data.normalize.mavis import normalize as mavis_normalize
from vrm.data.normalize.mm_eureka import normalize as mm_eureka_normalize
from vrm.data.normalize.tabmwp import normalize as tabmwp_normalize
from vrm.data.normalize.vision_r1_cold import normalize as vision_r1_cold_normalize
from vrm.data.normalize.we_math import normalize as we_math_normalize


def _img_qa_raw() -> dict:
    return {
        "id": "x",
        "image": "/tmp/x.png",
        "question": "Find x.",
        "answer": "42",
    }


@pytest.mark.parametrize(
    ("normalize", "source", "extras"),
    [
        (mavis_normalize, "mavis", {}),
        (mathv360k_normalize, "mathv360k", {}),
        (
            vision_r1_cold_normalize,
            "vision_r1_cold",
            {"response": "<think>x</think><answer>42</answer>"},
        ),
        (geo170k_normalize, "geo170k", {}),
        (chartqa_normalize, "chartqa", {"label": "42", "query": "What is shown?"}),
        (mm_eureka_normalize, "mm_eureka", {}),
        (geometry3k_normalize, "geometry3k", {"answer": "A", "choices": ["12", "24", "36"]}),
        (mathvista_normalize, "mathvista", {"pid": "p1"}),
        (we_math_normalize, "we_math", {"image_path": "/tmp/x.png", "image": None}),
        (geoqa_normalize, "geoqa", {"answer": "B"}),
        (tabmwp_normalize, "tabmwp", {"pid": "t1"}),
    ],
)
def test_basic_normalize_returns_record(normalize, source, extras):
    raw = _img_qa_raw() | extras
    rec = normalize(raw)
    assert rec is not None, f"{source} normalize returned None"
    assert rec.source == source
    assert rec.images, f"{source} record has no images"
    assert rec.answer
    assert rec.user_text()


@pytest.mark.parametrize(
    "normalize",
    [
        mavis_normalize,
        mathv360k_normalize,
        geo170k_normalize,
        mm_eureka_normalize,
        mathvista_normalize,
        tabmwp_normalize,
    ],
)
def test_drops_when_image_missing(normalize):
    raw = _img_qa_raw() | {"image": None}
    assert normalize(raw) is None


@pytest.mark.parametrize(
    "normalize",
    [
        mavis_normalize,
        mathv360k_normalize,
        geo170k_normalize,
        mathvista_normalize,
        tabmwp_normalize,
    ],
)
def test_drops_when_answer_missing(normalize):
    raw = _img_qa_raw() | {"answer": ""}
    assert normalize(raw) is None
