#!/usr/bin/env python3
"""Shared helpers for pairwise human-alignment annotation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRICS = [
    ("SF", "source_faithfulness", "Source faithfulness"),
    ("PER", "personalization", "Personalization"),
    ("APP", "applicability", "Applicability"),
    ("VID", "vividness", "Vividness"),
    ("LD", "logical_depth", "Logical depth"),
    ("FIT", "pq_fitness", "Practice question fitness"),
    ("GND", "pq_groundedness", "Practice question groundedness"),
    ("DIV", "pq_diversity", "Practice question diversity"),
    ("ANS", "pq_answer_quality", "Practice question answer quality"),
    ("CC", "pq_cross_concept", "Practice question cross-concept"),
]

METRIC_CODES = [code for code, _, _ in METRICS]
METRIC_BY_CODE = {code: {"key": key, "label": label} for code, key, label in METRICS}

PAIRWISE_COLUMNS = [
    "pair_id",
    "rater_id",
    *METRIC_CODES,
    "comment",
]

PREFERENCE_VALUES = {"A", "B", "tie"}
RUBRIC_VERSION = "benchmark_step3_pairwise_human_v1"

RUBRIC_MARKDOWN = """# DeepTutor Pairwise Human Alignment Rubric

You will compare two anonymous tutoring sessions, System A and System B. Both systems
respond to the same student profile, task, knowledge gaps, and source excerpts.

For each metric, choose:

- `A`: System A is better.
- `B`: System B is better.
- `tie`: The two systems are comparable, or the difference is too small to judge reliably.

Prefer a side when there is a clear quality difference. Use `tie` only when the evidence is
genuinely close or insufficient.

## Transcript metrics

- `SF` / source faithfulness: Which tutor is more faithful to the source excerpts and less likely to hallucinate or contradict them?
- `PER` / personalization: Which tutor better adapts to the student's profile, knowledge state, and confusion across the whole session?
- `APP` / applicability: Which tutor better helps the student make progress on the task and success criteria?
- `VID` / vividness: Which tutor gives more concrete, vivid, and example-supported explanations?
- `LD` / logical depth: Which tutor gives deeper, more coherent conceptual reasoning?

## Practice question metrics

- `FIT` / fitness: Which set of practice questions better fits the student and target gaps?
- `GND` / groundedness: Which set is more consistent with the source excerpts?
- `DIV` / diversity: Which set covers more varied angles rather than repeating one pattern?
- `ANS` / answer quality: Which set has better options, answers, and non-trivial distractors?
- `CC` / cross-concept: Which set better connects related concepts where appropriate?

Do not try to identify the system. The backend identity is intentionally hidden.
"""


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_preference(raw: Any) -> str | None:
    value = str(raw or "").strip().lower()
    if value in {"a", "system_a", "system a", "a better"}:
        return "A"
    if value in {"b", "system_b", "system b", "b better"}:
        return "B"
    if value in {"tie", "equal", "same", "draw", "相当", "平手"}:
        return "tie"
    return None
