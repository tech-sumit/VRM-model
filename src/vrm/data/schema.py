"""Canonical on-disk schema for a single VRM training/eval record.

Spec reference: VRM-7B_model_spec.md §3.3.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]
AnswerType = Literal["numeric", "multiple_choice", "latex_math", "span"]
VerifierName = Literal["exact_numeric", "normalize_choice", "math_equal", "span_match"]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Role
    content: str


class Record(BaseModel):
    """A single normalized training/eval record."""

    model_config = ConfigDict(extra="forbid")

    id: str
    images: list[str] = Field(default_factory=list, description="Local paths or s3://-style URIs")
    messages: list[Message]
    answer: str
    answer_type: AnswerType
    verifier: VerifierName
    tolerance: float = Field(default=0.0, ge=0.0)
    difficulty: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="base-model pass@K rate (0=hard, 1=easy)",
    )
    source: str = Field(description="Source dataset identifier, e.g. 'mavis', 'geoqa'")
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    def assistant_text(self) -> str:
        for m in reversed(self.messages):
            if m.role == "assistant":
                return m.content
        return ""

    def user_text(self) -> str:
        for m in self.messages:
            if m.role == "user":
                return m.content
        return ""
