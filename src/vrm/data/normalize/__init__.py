"""Registry of all source dataset normalizers.

Only datasets with working train splits and resolvable images are listed.
Sources temporarily removed pending further work:
  mathvista, we_math -- only testmini/test splits on HF; need alt source
                        for train-time data.
  tabmwp             -- Arietem/tabmwp has no image column (pure text tables).
  mm_eureka          -- schema audit stalled 2026-05-03; revisit in isolation.
"""

from __future__ import annotations

from vrm.data.normalize._base import NormalizeSpec
from vrm.data.normalize.chartqa import SPEC as chartqa_spec
from vrm.data.normalize.geo170k import SPEC as geo170k_spec
from vrm.data.normalize.geometry3k import SPEC as geometry3k_spec
from vrm.data.normalize.geoqa import SPEC as geoqa_spec
from vrm.data.normalize.mathv360k import SPEC as mathv360k_spec
from vrm.data.normalize.mavis import SPEC as mavis_spec
from vrm.data.normalize.vision_r1_cold import SPEC as vision_r1_cold_spec

REGISTRY: dict[str, NormalizeSpec] = {
    "mavis": mavis_spec,
    "mathv360k": mathv360k_spec,
    "vision_r1_cold": vision_r1_cold_spec,
    "geo170k": geo170k_spec,
    "chartqa": chartqa_spec,
    "geometry3k": geometry3k_spec,
    "geoqa": geoqa_spec,
}

__all__ = ["REGISTRY", "NormalizeSpec"]
