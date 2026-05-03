"""Geo170K normalizer (Luckyjhg/Geo170K).

LLaVA conversations format: {image, conversations}. Uses 'qa_tuning' split
(117K records) which is the real Q/A data; 'alignment' is pretraining
captions. Image paths are relative like "geoqa_plus/1000.png".
"""

from __future__ import annotations

from vrm.data.normalize._llava_convo import make_spec

SPEC = make_spec("geo170k", "Luckyjhg/Geo170K", split="qa_tuning")
normalize = SPEC.normalize
