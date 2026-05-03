# Sub-plan 03 — Data schema & verifiers

> Index: [`../plan.md`](../plan.md) · Depends on: [00 bootstrap](00-bootstrap.md) · Status: ☐

**Goal:** Lock the on-disk record schema (Pydantic) and implement the four deterministic verifiers — numeric, multiple-choice, latex-math, span — used by both data filtering (Stage 0) and RL reward (Stage 2). Per spec §3.3, no LLM-judge.

**Architecture:** One `Record` Pydantic model representing a single training example. A `verifier_registry` dict maps `verifier` field values to functions of signature `(prediction: str, gold: dict) -> float`. Format-aware reward composition (`0.1·format + 0.9·accuracy`) lives in [sub-plan 07](07-stage2-grpo.md), but the per-verifier accuracy primitives live here.

**Tech Stack:** pydantic 2 · sympy · regex.

---

### Task 1: `Record` schema (`vrm/data/schema.py`)

**Files:**
- Create: `projects/VRM-model/src/vrm/data/schema.py`
- Create: `projects/VRM-model/tests/unit/test_schema.py`

- [ ] **Step 1: Write the failing test**

`projects/VRM-model/tests/unit/test_schema.py`:

```python
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
            {"role": "assistant", "content": "<think>...</think><answer>72</answer>"},
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
```

- [ ] **Step 2: Run (FAILS)**

```bash
make test
```

- [ ] **Step 3: Implement**

`projects/VRM-model/src/vrm/data/schema.py`:

```python
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
    difficulty: float | None = Field(default=None, ge=0.0, le=1.0,
                                      description="base-model pass@K rate (0=hard, 1=easy)")
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
```

- [ ] **Step 4: Run (PASSES)**

```bash
make test && make typecheck
```

- [ ] **Step 5: Commit**

```bash
git add projects/VRM-model/src/vrm/data/schema.py projects/VRM-model/tests/unit/test_schema.py
git commit -m "vrm: add Record schema (Pydantic) per spec §3.3"
```

---

### Task 2: Numeric verifier (`vrm/data/verifiers/numeric.py`)

**Files:**
- Create: `projects/VRM-model/src/vrm/data/verifiers/numeric.py`
- Create: `projects/VRM-model/tests/unit/test_verifiers_numeric.py`

- [ ] **Step 1: Failing test**

`projects/VRM-model/tests/unit/test_verifiers_numeric.py`:

```python
import pytest

from vrm.data.verifiers.numeric import exact_numeric


@pytest.mark.parametrize("pred, gold, tol, expected", [
    ("42", "42", 0.0, 1.0),
    ("42.0", "42", 0.0, 1.0),
    ("3.14", "3.14159", 0.01, 1.0),
    ("3.14", "3.14159", 0.001, 0.0),
    (" 42 ", "42", 0.0, 1.0),
    ("$42$", "42", 0.0, 1.0),
    ("42%", "42", 0.0, 1.0),
    ("1,000", "1000", 0.0, 1.0),
    ("not a number", "42", 0.0, 0.0),
    ("", "42", 0.0, 0.0),
    ("42", "", 0.0, 0.0),
    ("-3.5", "-3.5", 0.0, 1.0),
    ("1e3", "1000", 0.0, 1.0),
    ("\\frac{1}{2}", "0.5", 0.001, 1.0),
    ("0.5", "1/2", 0.001, 1.0),
])
def test_exact_numeric(pred, gold, tol, expected):
    assert exact_numeric(pred, {"answer": gold, "tolerance": tol}) == expected
```

- [ ] **Step 2: Run (FAILS)**

- [ ] **Step 3: Implement**

`projects/VRM-model/src/vrm/data/verifiers/numeric.py`:

```python
"""Numeric verifier with permissive normalization.

Handles dollar signs, percent signs, commas, latex fractions, plain fractions,
exponential notation, leading/trailing whitespace.
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

_LATEX_FRAC = re.compile(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}")
_PLAIN_FRAC = re.compile(r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$")


def _coerce(s: str) -> float | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    s = s.strip("$ ").rstrip("%").replace(",", "")
    m = _LATEX_FRAC.search(s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    m = _PLAIN_FRAC.match(s)
    if m:
        try:
            return float(Fraction(m.group(1)) / Fraction(m.group(2)))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def exact_numeric(prediction: str, gold: dict[str, Any]) -> float:
    """Returns 1.0 if numeric prediction matches gold within tolerance, else 0.0."""
    p = _coerce(prediction)
    g = _coerce(str(gold.get("answer", "")))
    if p is None or g is None:
        return 0.0
    tol = float(gold.get("tolerance", 0.0))
    return 1.0 if abs(p - g) <= tol else 0.0
```

- [ ] **Step 4: Run (PASSES)**

- [ ] **Step 5: Commit**

```bash
git add projects/VRM-model/src/vrm/data/verifiers/numeric.py projects/VRM-model/tests/unit/test_verifiers_numeric.py
git commit -m "vrm: add numeric verifier with permissive normalization"
```

---

### Task 3: Multiple-choice verifier

**Files:**
- Create: `projects/VRM-model/src/vrm/data/verifiers/multiple_choice.py`
- Create: `projects/VRM-model/tests/unit/test_verifiers_mc.py`

- [ ] **Step 1: Failing test**

```python
# projects/VRM-model/tests/unit/test_verifiers_mc.py
import pytest

from vrm.data.verifiers.multiple_choice import normalize_choice


@pytest.mark.parametrize("pred, gold, expected", [
    ("A", "A", 1.0),
    ("a", "A", 1.0),
    ("(A)", "A", 1.0),
    ("A.", "A", 1.0),
    ("A)", "A", 1.0),
    ("Answer: A", "A", 1.0),
    ("The answer is A.", "A", 1.0),
    ("B", "A", 0.0),
    ("None", "A", 0.0),
    ("", "A", 0.0),
])
def test_normalize_choice(pred, gold, expected):
    assert normalize_choice(pred, {"answer": gold}) == expected
```

- [ ] **Step 2: Implement**

`projects/VRM-model/src/vrm/data/verifiers/multiple_choice.py`:

```python
"""Multiple-choice verifier. Extracts a single uppercase letter A-J from a free-form response."""
from __future__ import annotations

import re
from typing import Any

_LETTER_RE = re.compile(r"\b([A-J])\b")


def _extract(text: str) -> str | None:
    if not text:
        return None
    text = text.strip()
    # Strip wrapping punctuation
    text = text.lstrip("(").rstrip(").")
    # Direct single-letter
    if len(text) == 1 and text.upper() in "ABCDEFGHIJ":
        return text.upper()
    # "Answer: X" / "The answer is X" / "X is correct"
    upper = text.upper()
    m = re.search(r"ANSWER[:\s]+\(?([A-J])\)?", upper)
    if m:
        return m.group(1)
    m = re.search(r"\b(?:THE\s+ANSWER\s+IS|FINAL\s+ANSWER[:\s])\s*\(?([A-J])\)?", upper)
    if m:
        return m.group(1)
    # Last resort: first standalone letter A-J
    m = _LETTER_RE.search(upper)
    return m.group(1) if m else None


def normalize_choice(prediction: str, gold: dict[str, Any]) -> float:
    p = _extract(prediction)
    g = (gold.get("answer") or "").strip().upper()
    if not p or not g:
        return 0.0
    return 1.0 if p == g else 0.0
```

- [ ] **Step 3: Run, commit**

```bash
make test
git add projects/VRM-model/src/vrm/data/verifiers/multiple_choice.py projects/VRM-model/tests/unit/test_verifiers_mc.py
git commit -m "vrm: add multiple-choice verifier with answer-extraction heuristics"
```

---

### Task 4: LaTeX-math verifier (sympy)

**Files:**
- Create: `projects/VRM-model/src/vrm/data/verifiers/latex_math.py`
- Create: `projects/VRM-model/tests/unit/test_verifiers_latex.py`

- [ ] **Step 1: Failing test**

```python
# projects/VRM-model/tests/unit/test_verifiers_latex.py
import pytest

from vrm.data.verifiers.latex_math import math_equal


@pytest.mark.parametrize("pred, gold, expected", [
    ("x^2 + 2x + 1", "(x+1)^2", 1.0),
    ("\\frac{1}{2}", "0.5", 1.0),
    ("\\sqrt{4}", "2", 1.0),
    ("\\pi", "3.14159265358979", 0.0),  # symbolic vs numeric — should not match without simplify
    ("2x + 3", "3 + 2x", 1.0),
    ("x^2", "x**2", 1.0),
    ("not math", "x+1", 0.0),
])
def test_math_equal(pred, gold, expected):
    assert math_equal(pred, {"answer": gold}) == expected
```

- [ ] **Step 2: Implement**

`projects/VRM-model/src/vrm/data/verifiers/latex_math.py`:

```python
"""LaTeX-math verifier using sympy.

`math_equal(pred, gold)` returns 1.0 iff `simplify(pred - gold) == 0`
after parsing both via sympy's latex / sympify with reasonable preprocessing.
"""
from __future__ import annotations

import re
from typing import Any

from sympy import simplify, sympify
from sympy.parsing.latex import parse_latex


_LATEX_MARKERS = ("\\frac", "\\sqrt", "\\sum", "\\int", "\\pi", "\\theta", "^{", "_{")


def _looks_latex(s: str) -> bool:
    return any(m in s for m in _LATEX_MARKERS)


def _to_sympy(s: str):
    s = s.strip().strip("$ ")
    if not s:
        return None
    try:
        if _looks_latex(s):
            return parse_latex(s)
        # Common ASCII-math conventions
        s = s.replace("^", "**")
        return sympify(s)
    except Exception:  # noqa: BLE001
        return None


def math_equal(prediction: str, gold: dict[str, Any]) -> float:
    p = _to_sympy(prediction)
    g = _to_sympy(str(gold.get("answer", "")))
    if p is None or g is None:
        return 0.0
    try:
        diff = simplify(p - g)
        return 1.0 if diff == 0 else 0.0
    except Exception:  # noqa: BLE001
        return 0.0
```

- [ ] **Step 3: Run (note: sympy.parsing.latex requires `antlr4-python3-runtime`; add as transitive)**

If pytest fails with `ImportError: antlr4`, add `antlr4-python3-runtime>=4.11` to `[project.dependencies]` in `pyproject.toml`, then `uv sync --extra dev`.

- [ ] **Step 4: Commit**

```bash
git add projects/VRM-model/src/vrm/data/verifiers/latex_math.py projects/VRM-model/tests/unit/test_verifiers_latex.py projects/VRM-model/pyproject.toml projects/VRM-model/uv.lock
git commit -m "vrm: add latex-math verifier (sympy.simplify-based equality)"
```

---

### Task 5: Span verifier (substring / set-of-spans match)

**Files:**
- Create: `projects/VRM-model/src/vrm/data/verifiers/span.py`
- Create: `projects/VRM-model/tests/unit/test_verifiers_span.py`

- [ ] **Step 1: Failing test**

```python
# projects/VRM-model/tests/unit/test_verifiers_span.py
import pytest

from vrm.data.verifiers.span import span_match


@pytest.mark.parametrize("pred, gold, expected", [
    ("the answer is paris", "Paris", 1.0),
    ("PARIS", "paris", 1.0),
    ("london", "Paris", 0.0),
    ("paris, france", "paris", 1.0),
    ("", "paris", 0.0),
])
def test_span_match(pred, gold, expected):
    assert span_match(pred, {"answer": gold}) == expected
```

- [ ] **Step 2: Implement**

`projects/VRM-model/src/vrm/data/verifiers/span.py`:

```python
"""Span verifier — case-insensitive substring containment.

For ChartQA / span-style answers where the model emits a sentence and the
gold is a short noun-phrase. Returns 1.0 iff `gold.lower() in pred.lower()`.
"""
from __future__ import annotations

from typing import Any


def span_match(prediction: str, gold: dict[str, Any]) -> float:
    g = (gold.get("answer") or "").strip().lower()
    p = (prediction or "").strip().lower()
    if not g or not p:
        return 0.0
    return 1.0 if g in p else 0.0
```

- [ ] **Step 3: Run, commit**

```bash
make test
git add projects/VRM-model/src/vrm/data/verifiers/span.py projects/VRM-model/tests/unit/test_verifiers_span.py
git commit -m "vrm: add span verifier (case-insensitive substring containment)"
```

---

### Task 6: Verifier registry + format check + composite reward

**Files:**
- Create: `projects/VRM-model/src/vrm/data/verifiers/__init__.py`
- Create: `projects/VRM-model/src/vrm/data/verifiers/format.py`
- Create: `projects/VRM-model/tests/unit/test_verifiers_format.py`
- Create: `projects/VRM-model/tests/unit/test_verifiers_registry.py`

- [ ] **Step 1: Format check (extracts `<think>...</think><answer>...</answer>`)**

`projects/VRM-model/src/vrm/data/verifiers/format.py`:

```python
"""Format verifier: per spec §3.3, response must be `<think>...</think><answer>...</answer>`.

Used both as part of the composite reward and to extract the model's predicted answer.
"""
from __future__ import annotations

import re

_FMT = re.compile(
    r"<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>",
    re.DOTALL,
)
MIN_THINK_TOKENS = 50  # rough whitespace-token count to discourage reward hacking


def has_valid_format(response: str) -> bool:
    m = _FMT.search(response or "")
    if not m:
        return False
    think_tokens = len(m.group("think").split())
    return think_tokens >= MIN_THINK_TOKENS


def extract_answer(response: str) -> str:
    m = _FMT.search(response or "")
    return m.group("answer").strip() if m else ""
```

`projects/VRM-model/tests/unit/test_verifiers_format.py`:

```python
from vrm.data.verifiers.format import extract_answer, has_valid_format


def _think(n=60):
    return " ".join(["x"] * n)


def test_valid_format_with_long_think():
    r = f"<think>{_think()}</think><answer>72</answer>"
    assert has_valid_format(r)
    assert extract_answer(r) == "72"


def test_invalid_when_think_too_short():
    r = "<think>short</think><answer>72</answer>"
    assert not has_valid_format(r)


def test_invalid_when_no_tags():
    assert not has_valid_format("just an answer 72")
    assert extract_answer("just an answer 72") == ""


def test_handles_newlines_and_whitespace():
    r = f"<think>\n{_think()}\n</think>\n<answer> 72 </answer>"
    assert has_valid_format(r)
    assert extract_answer(r) == "72"
```

- [ ] **Step 2: Registry that maps verifier names to functions**

`projects/VRM-model/src/vrm/data/verifiers/__init__.py`:

```python
"""Verifier registry. Use `score(record, prediction)` to compute composite reward."""
from __future__ import annotations

from typing import Any, Callable

from vrm.data.verifiers.format import extract_answer, has_valid_format
from vrm.data.verifiers.latex_math import math_equal
from vrm.data.verifiers.multiple_choice import normalize_choice
from vrm.data.verifiers.numeric import exact_numeric
from vrm.data.verifiers.span import span_match

VerifierFn = Callable[[str, dict[str, Any]], float]

REGISTRY: dict[str, VerifierFn] = {
    "exact_numeric": exact_numeric,
    "normalize_choice": normalize_choice,
    "math_equal": math_equal,
    "span_match": span_match,
}


def score(record_or_gold: dict[str, Any], prediction: str, *, format_weight: float = 0.1) -> dict[str, float]:
    """Composite reward per spec §3.3:
        total = format_weight * format + (1 - format_weight) * accuracy
    `record_or_gold` is a dict with at least keys 'verifier', 'answer', and optional 'tolerance'.
    """
    fmt = 1.0 if has_valid_format(prediction) else 0.0
    if fmt == 0.0:
        return {"format": 0.0, "accuracy": 0.0, "total": 0.0}
    pred_answer = extract_answer(prediction)
    name = record_or_gold.get("verifier")
    fn = REGISTRY.get(name)
    if fn is None:
        return {"format": fmt, "accuracy": 0.0, "total": format_weight * fmt}
    acc = fn(pred_answer, record_or_gold)
    total = format_weight * fmt + (1.0 - format_weight) * acc
    return {"format": fmt, "accuracy": acc, "total": total}
```

- [ ] **Step 3: Tests for registry composition**

`projects/VRM-model/tests/unit/test_verifiers_registry.py`:

```python
from vrm.data.verifiers import score


def _good_response(answer="72"):
    think = " ".join(["x"] * 60)
    return f"<think>{think}</think><answer>{answer}</answer>"


def test_score_correct_numeric():
    res = score({"verifier": "exact_numeric", "answer": "72", "tolerance": 0.0}, _good_response("72"))
    assert res == {"format": 1.0, "accuracy": 1.0, "total": 1.0}


def test_score_wrong_numeric_keeps_format_credit():
    res = score({"verifier": "exact_numeric", "answer": "72", "tolerance": 0.0}, _good_response("73"))
    assert res == {"format": 1.0, "accuracy": 0.0, "total": 0.1}


def test_score_no_format_zeroes_total():
    res = score({"verifier": "exact_numeric", "answer": "72", "tolerance": 0.0}, "73")
    assert res == {"format": 0.0, "accuracy": 0.0, "total": 0.0}


def test_score_unknown_verifier_returns_format_only():
    res = score({"verifier": "unknown", "answer": "x"}, _good_response())
    assert res["accuracy"] == 0.0
    assert res["format"] == 1.0
```

- [ ] **Step 4: Run, commit**

```bash
make test && make typecheck
git add projects/VRM-model/src/vrm/data/verifiers/__init__.py \
        projects/VRM-model/src/vrm/data/verifiers/format.py \
        projects/VRM-model/tests/unit/test_verifiers_format.py \
        projects/VRM-model/tests/unit/test_verifiers_registry.py
git commit -m "vrm: add format check + verifier registry with composite reward (spec §3.3)"
```

---

## Done when

- [ ] `make test` shows ~30+ passing tests covering the schema and 5 verifiers.
- [ ] `make typecheck` clean.
- [ ] Sub-plan 04 (data normalize) can import `from vrm.data.schema import Record`.
- [ ] Sub-plan 07 (Stage 2 GRPO) can import `from vrm.data.verifiers import score` for the reward function.
