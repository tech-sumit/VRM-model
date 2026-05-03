"""Vision-R1-cold normalizer (Osilly/Vision-R1-cold).

The repo ships two JSONs with mismatched columns
(vision_r1_llava_cot_full.json has {id,image,conversations};
 vision_r1_mulberry_sft_full.json has {images,conversations}).
Pinning to the llava_cot variant via data_files avoids the schema clash
that kills datasets.load_dataset when it tries to merge both.
"""

from __future__ import annotations

from vrm.data.normalize._base import NormalizeSpec
from vrm.data.normalize._llava_convo import make_normalizer

SPEC = NormalizeSpec(
    hf_id="Osilly/Vision-R1-cold",
    split="train",
    normalize=make_normalizer("vision_r1_cold"),
    default_verifier="span_match",
    data_files="vision_r1_llava_cot_full.json",
)
normalize = SPEC.normalize
