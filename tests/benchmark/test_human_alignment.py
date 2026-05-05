from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark.human_alignment.common import METRIC_CODES
from benchmark.human_alignment.export_annotations import export_annotation_package
from benchmark.human_alignment.plot_alignment import build_svg
from benchmark.human_alignment.summarize_annotations import summarize_annotations


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _fake_entry(entry_id: str) -> dict:
    return {
        "entry_id": entry_id,
        "kb_name": "Calculus",
        "profile": {
            "profile_id": "p1",
            "personality": "curious",
            "education_background": "undergraduate",
            "learning_purpose": "learn derivatives",
            "knowledge_state": {
                "known_well": ["limits"],
                "partially_known": ["derivatives"],
                "unknown": ["chain rule"],
            },
        },
        "task": {
            "title": f"Task {entry_id}",
            "description": "Explain the chain rule.",
            "success_criteria": "Student can apply the chain rule.",
            "target_gaps": ["g1"],
        },
        "gaps": [
            {
                "gap_id": "g1",
                "target_concept": "Chain rule",
                "gap_type": "missing",
                "description": "Does not know nested derivatives.",
                "source_pages": [1],
            }
        ],
        "source_content": {"1": "The chain rule differentiates composite functions."},
    }


def _write_transcript(output_root: Path, backend: str, label: str) -> None:
    _write_json(
        output_root / "transcripts" / "Calculus" / backend / "p1.json",
        {
            "profile_id": "p1",
            "sessions": [
                {
                    "entry_id": "e1",
                    "entry": _fake_entry("e1"),
                    "transcript": [
                        {"role": "student", "content": "I do not get this."},
                        {"role": "tutor", "content": f"{label}: Let's use a nested function."},
                    ],
                    "practice_questions": [f"{label} Q1"],
                },
                {
                    "entry_id": "e2",
                    "entry": _fake_entry("e2"),
                    "transcript": [
                        {"role": "student", "content": "Another example?"},
                        {"role": "tutor", "content": f"{label}: Try sin(x^2)."},
                    ],
                    "practice_questions": [f"{label} Q2"],
                },
            ],
        },
    )


def _metric_summary(value: float) -> dict:
    return {
        "source_faithfulness": {"avg_score": value},
        "teaching_quality": {
            "personalization": {"avg": value},
            "applicability": {"avg": value},
            "vividness": {"avg": value},
            "logical_depth": {"avg": value},
        },
        "practice_questions": {
            "summary": {
                "avg_fitness": value,
                "avg_groundedness": value,
                "avg_diversity": value,
                "avg_answer_quality": value,
                "avg_cross_concept": value,
            }
        },
    }


def _write_eval(output_root: Path, backend: str, e1_score: float, e2_score: float) -> Path:
    eval_path = output_root / "evaluations" / "Calculus" / backend / "p1_eval.json"
    _write_json(
        eval_path,
        {
            "profile_id": "p1",
            "sessions": [
                {"entry_id": "e1", "metrics": _metric_summary(e1_score)},
                {"entry_id": "e2", "metrics": _metric_summary(e2_score)},
            ],
        },
    )
    return eval_path


def test_export_pairwise_annotations_matches_sessions_and_hides_backend(tmp_path: Path) -> None:
    output_root = tmp_path / "bench"
    _write_transcript(output_root, "deep_tutor", "DT")
    _write_transcript(output_root, "mock", "MOCK")

    manifest = export_annotation_package(
        output_root=output_root,
        kb_names=["Calculus"],
        output_dir=tmp_path / "human",
        target_backend="deep_tutor",
        baseline_backend="mock",
        seed=7,
    )

    rows = [
        json.loads(line)
        for line in Path(manifest["package_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["pair_id"].startswith("hp_")
    assert "backend" not in json.dumps(rows)
    assert set(rows[0]) >= {"profile", "task", "gaps", "system_a", "system_b"}
    assert rows[0]["gaps"][0]["source_excerpts"]["1"]

    key = json.loads(Path(manifest["annotation_key_path"]).read_text(encoding="utf-8"))
    assert {item["target_backend"] for item in key["items"]} == {"deep_tutor"}
    assert {item["baseline_backend"] for item in key["items"]} == {"mock"}
    assert {item["session_index"] for item in key["items"]} == {1, 2}
    assert {item["system_a_backend"] for item in key["items"]} | {item["system_b_backend"] for item in key["items"]} == {"deep_tutor", "mock"}

    template_header = Path(manifest["annotation_template_path"]).read_text(encoding="utf-8").splitlines()[0]
    assert template_header == "pair_id,rater_id,SF,PER,APP,VID,LD,FIT,GND,DIV,ANS,CC,comment"
    assert Path(manifest["review_ui_path"]).exists()


def test_export_pairwise_annotations_removes_duplicate_practice_turn(tmp_path: Path) -> None:
    output_root = tmp_path / "bench"
    practice_questions = [
        "Question 1: What does the chain rule compute?",
        "Question 2: Differentiate sin(x^2).",
    ]
    for backend, label in [("deep_tutor", "DT"), ("mock", "MOCK")]:
        _write_json(
            output_root / "transcripts" / "Calculus" / backend / "p1.json",
            {
                "profile_id": "p1",
                "sessions": [
                    {
                        "entry_id": "e1",
                        "entry": _fake_entry("e1"),
                        "transcript": [
                            {"role": "student", "content": "I do not get this."},
                            {"role": "tutor", "content": f"{label}: Let's use a nested function."},
                            {
                                "role": "tutor",
                                "content": "\n\n".join(practice_questions),
                            },
                        ],
                        "practice_questions": practice_questions,
                    },
                ],
            },
        )

    manifest = export_annotation_package(
        output_root=output_root,
        kb_names=["Calculus"],
        output_dir=tmp_path / "human",
        target_backend="deep_tutor",
        baseline_backend="mock",
        seed=7,
    )

    row = json.loads(Path(manifest["package_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert row["system_a"]["turn_count"]["dialog_messages"] == 2
    assert row["system_b"]["turn_count"]["dialog_messages"] == 2
    assert len(row["system_a"]["dialog"]) == 2
    assert len(row["system_b"]["dialog"]) == 2
    assert row["system_a"]["practice_questions"] == practice_questions
    assert row["system_b"]["practice_questions"] == practice_questions


def test_summarize_pairwise_annotations_majority_and_llm_threshold(tmp_path: Path) -> None:
    output_root = tmp_path / "bench"
    _write_eval(output_root, "deep_tutor", e1_score=4.8, e2_score=4.1)
    _write_eval(output_root, "mock", e1_score=3.9, e2_score=4.0)

    key_path = tmp_path / "human" / "annotation_key.json"
    _write_json(
        key_path,
        {
            "items": [
                {
                    "pair_id": "hp_000001",
                    "kb_name": "Calculus",
                    "profile_id": "p1",
                    "entry_id": "e1",
                    "session_index": 1,
                    "target_backend": "deep_tutor",
                    "baseline_backend": "mock",
                    "system_a_backend": "deep_tutor",
                    "system_b_backend": "mock",
                    "system_a_evaluation_path": str(output_root / "evaluations" / "Calculus" / "deep_tutor" / "p1_eval.json"),
                    "system_b_evaluation_path": str(output_root / "evaluations" / "Calculus" / "mock" / "p1_eval.json"),
                },
                {
                    "pair_id": "hp_000002",
                    "kb_name": "Calculus",
                    "profile_id": "p1",
                    "entry_id": "e2",
                    "session_index": 2,
                    "target_backend": "deep_tutor",
                    "baseline_backend": "mock",
                    "system_a_backend": "mock",
                    "system_b_backend": "deep_tutor",
                    "system_a_evaluation_path": str(output_root / "evaluations" / "Calculus" / "mock" / "p1_eval.json"),
                    "system_b_evaluation_path": str(output_root / "evaluations" / "Calculus" / "deep_tutor" / "p1_eval.json"),
                },
            ]
        },
    )

    annotations_path = tmp_path / "human" / "completed.csv"
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    with open(annotations_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "rater_id", *METRIC_CODES, "comment"])
        writer.writeheader()
        # Pair 1: majority prefers A, which is DeepTutor; LLM also prefers DeepTutor.
        for rater, pref in [("r1", "A"), ("r2", "A"), ("r3", "B")]:
            row = {"pair_id": "hp_000001", "rater_id": rater, "comment": ""}
            row.update({code: pref for code in METRIC_CODES})
            writer.writerow(row)
        # Pair 2: human majority prefers B, which is DeepTutor; LLM delta is 0.1 <= 0.25, so tie.
        for rater, pref in [("r1", "B"), ("r2", "B"), ("r3", "tie")]:
            row = {"pair_id": "hp_000002", "rater_id": rater, "comment": ""}
            row.update({code: pref for code in METRIC_CODES})
            writer.writerow(row)

    summary = summarize_annotations(
        annotations_path=annotations_path,
        key_path=key_path,
        output_path=tmp_path / "human" / "summary.json",
        tie_threshold=0.25,
    )

    metric = summary["metrics"]["SF"]
    assert summary["num_annotation_rows"] == 6
    assert summary["num_pairs_with_human_labels"] == 2
    assert summary["num_raters"] == 3
    assert metric["n"] == 2
    assert metric["human_target_preference_rate"] == 1.0
    assert metric["llm_target_preference_rate"] == 0.5
    assert metric["llm_tie_rate"] == 0.5
    assert metric["agreement_rate"] == 0.5
    assert metric["counts"]["human"] == {"target": 2}
    assert summary["pairs"][0]["metrics"]["SF"]["human_backend_preference"] == "target"
    assert summary["pairs"][1]["metrics"]["SF"]["llm_backend_preference"] == "tie"
    assert (tmp_path / "human" / "summary.md").exists()


def test_plot_alignment_builds_svg_from_summary() -> None:
    summary = {
        "metrics": {
            code: {
                "human_target_preference_rate": 0.6,
                "human_tie_rate": 0.2,
                "human_baseline_preference_rate": 0.2,
                "llm_target_preference_rate": 0.7,
                "llm_tie_rate": 0.1,
                "llm_baseline_preference_rate": 0.2,
            }
            for code in METRIC_CODES
        }
    }

    svg = build_svg(summary, "Human vs. LLM Preference Alignment")

    assert svg.startswith("<svg")
    assert "Human vs. LLM Preference Alignment" in svg
    assert "DeepTutor preferred" in svg
    assert "Mock preferred" in svg
    assert ">60<" in svg
    assert ">70<" in svg
