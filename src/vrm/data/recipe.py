"""Recipe schema and loader."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class SourceCap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    cap: int


class DistillationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    concurrency: int = 16
    ensemble: list[str] = []


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    target_size: int
    difficulty_lo: float
    difficulty_hi: float
    sources: list[SourceCap]
    distillation: DistillationConfig


def load_recipe(path: Path) -> Recipe:
    return Recipe.model_validate(yaml.safe_load(Path(path).read_text()))
