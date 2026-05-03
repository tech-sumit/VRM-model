"""Registry of all source dataset normalizers."""

from __future__ import annotations

from vrm.data.normalize._base import NormalizeSpec
from vrm.data.normalize.chartqa import SPEC as chartqa_spec
from vrm.data.normalize.geo170k import SPEC as geo170k_spec
from vrm.data.normalize.geometry3k import SPEC as geometry3k_spec
from vrm.data.normalize.geoqa import SPEC as geoqa_spec
from vrm.data.normalize.mathv360k import SPEC as mathv360k_spec
from vrm.data.normalize.mathvista import SPEC as mathvista_spec
from vrm.data.normalize.mavis import SPEC as mavis_spec
from vrm.data.normalize.mm_eureka import SPEC as mm_eureka_spec
from vrm.data.normalize.tabmwp import SPEC as tabmwp_spec
from vrm.data.normalize.vision_r1_cold import SPEC as vision_r1_cold_spec
from vrm.data.normalize.we_math import SPEC as we_math_spec

REGISTRY: dict[str, NormalizeSpec] = {
    "mavis": mavis_spec,
    "mathv360k": mathv360k_spec,
    "vision_r1_cold": vision_r1_cold_spec,
    "geo170k": geo170k_spec,
    "chartqa": chartqa_spec,
    "mm_eureka": mm_eureka_spec,
    "geometry3k": geometry3k_spec,
    "mathvista": mathvista_spec,
    "we_math": we_math_spec,
    "geoqa": geoqa_spec,
    "tabmwp": tabmwp_spec,
}

__all__ = ["REGISTRY", "NormalizeSpec"]
