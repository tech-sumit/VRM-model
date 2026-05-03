"""MathV360K normalizer (Zhiqiang007/MathV360K).

LLaVA conversations format: {id, image, conversations}. The image is a
relative path like "DVQA/images/bar_train_00134926.png" which refers into
a companion ZIP archive on the dataset repo; we keep the relative path
and defer resolution to train-time.
"""

from __future__ import annotations

from vrm.data.normalize._llava_convo import make_spec

SPEC = make_spec("mathv360k", "Zhiqiang007/MathV360K", split="train")
normalize = SPEC.normalize
