#!/usr/bin/env python3
"""
Export blinded pairwise human-alignment annotation packages.

The exporter matches sessions from a target backend and a baseline backend under
the same KB/profile/entry/session, randomizes which backend appears as System A,
and writes a public JSONL package plus a private key.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import random
import sys
from typing import Any

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.human_alignment.common import (
    PAIRWISE_COLUMNS,
    RUBRIC_MARKDOWN,
    RUBRIC_VERSION,
    read_json,
    write_json,
    write_jsonl,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmark" / "data" / "bench_pipeline"
REVIEW_UI_SOURCE = Path(__file__).with_name("review_ui.html")


def _parse_names(raw: str) -> list[str]:
    return sorted(set(n.strip() for n in raw.split(",") if n.strip()))


def _short_text(text: Any, max_chars: int) -> str:
    value = str(text or "").strip()
    return value[:max_chars] + ("..." if len(value) > max_chars else "")


def _format_profile(profile: dict[str, Any]) -> dict[str, Any]:
    ks = profile.get("knowledge_state", {}) or {}
    return {
        "profile_id": profile.get("profile_id", ""),
        "personality": profile.get("personality", ""),
        "education_background": profile.get("education_background", ""),
        "learning_purpose": profile.get("learning_purpose", ""),
        "known_well": ks.get("known_well", []),
        "partially_known": ks.get("partially_known", []),
        "unknown": ks.get("unknown", []),
        "beliefs": profile.get("beliefs", ""),
    }


def _source_excerpt_by_page(source_content: dict[str, Any] | None, max_chars: int) -> dict[str, str]:
    if not isinstance(source_content, dict):
        return {}
    excerpts: dict[str, str] = {}
    for page, text in sorted(source_content.items(), key=lambda item: str(item[0])):
        excerpt = _short_text(text, max_chars)
        if excerpt:
            excerpts[str(page)] = excerpt
    return excerpts


def _format_gaps(
    gaps: list[dict[str, Any]],
    source_content: dict[str, Any] | None,
    max_chars: int,
) -> list[dict[str, Any]]:
    source_by_page = _source_excerpt_by_page(source_content, max_chars)
    formatted = []
    for gap in gaps:
        pages = [str(p) for p in gap.get("source_pages", [])]
        formatted.append(
            {
                "gap_id": gap.get("gap_id", ""),
                "target_concept": gap.get("target_concept", ""),
                "gap_type": gap.get("gap_type", ""),
                "description": gap.get("description", ""),
                "manifestation": gap.get("manifestation", ""),
                "correct_understanding": gap.get("correct_understanding", ""),
                "source_pages": pages,
                "source_excerpts": {p: source_by_page[p] for p in pages if p in source_by_page},
            }
        )
    return formatted


def _dialog_messages(transcript: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages = []
    for msg in transcript:
        role = msg.get("role")
        if role in {"student", "tutor"}:
            messages.append({"role": role, "content": str(msg.get("content", ""))})
    return messages


def _session_match_key(kb_name: str, profile_id: str, entry_id: str, session_index: int) -> tuple[str, str, str]:
    session_id = entry_id or f"session_{session_index}"
    return (kb_name, profile_id, session_id)


def _session_records_for_backend(
    *,
    output_root: Path,
    kb_name: str,
    backend: str,
    source_chars: int,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    backend_dir = output_root / "transcripts" / kb_name / backend
    if not backend_dir.exists():
        return records

    for transcript_path in sorted(backend_dir.glob("*.json")):
        data = read_json(transcript_path)
        profile_id_from_file = transcript_path.stem
        eval_path = output_root / "evaluations" / kb_name / backend / f"{profile_id_from_file}_eval.json"
        raw_sessions = data["sessions"] if isinstance(data, dict) and isinstance(data.get("sessions"), list) else [data]

        for idx, session in enumerate(raw_sessions, start=1):
            if not isinstance(session, dict):
                continue
            entry = session.get("entry") or data.get("entry", {}) if isinstance(data, dict) else {}
            profile = entry.get("profile", {}) or {}
            task = entry.get("task", {}) or {}
            profile_id = profile.get("profile_id") or profile_id_from_file
            entry_id = session.get("entry_id") or entry.get("entry_id") or ""
            key = _session_match_key(kb_name, str(profile_id), str(entry_id), idx)
            transcript = session.get("transcript", []) or []
            practice_questions = session.get("practice_questions", []) or []
            records[key] = {
                "kb_name": kb_name,
                "backend": backend,
                "profile_id": str(profile_id),
                "entry_id": str(entry_id),
                "session_index": idx,
                "transcript_path": str(transcript_path),
                "evaluation_path": str(eval_path),
                "profile": _format_profile(profile),
                "task": {
                    "title": task.get("title", ""),
                    "description": task.get("description", ""),
                    "success_criteria": task.get("success_criteria", ""),
                    "target_gaps": task.get("target_gaps", []),
                },
                "gaps": _format_gaps(entry.get("gaps", []) or [], entry.get("source_content"), source_chars),
                "dialog": _dialog_messages(transcript),
                "practice_questions": practice_questions,
                "turn_count": {
                    "dialog_messages": len(_dialog_messages(transcript)),
                    "practice_questions": len(practice_questions),
                },
            }
    return records


def _write_template(path: Path, pair_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIRWISE_COLUMNS)
        writer.writeheader()
        for pair_id in pair_ids:
            writer.writerow({"pair_id": pair_id})


def _public_system(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dialog": record["dialog"],
        "practice_questions": record["practice_questions"],
        "turn_count": record["turn_count"],
    }


def export_annotation_package(
    *,
    output_root: Path,
    kb_names: list[str],
    output_dir: Path,
    target_backend: str = "deep_tutor",
    baseline_backend: str = "mock",
    source_chars: int = 1500,
    limit_pairs: int = 0,
    seed: int = 13,
) -> dict[str, Any]:
    rng = random.Random(seed)
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    unmatched: list[dict[str, Any]] = []

    for kb_name in kb_names:
        target_records = _session_records_for_backend(
            output_root=output_root,
            kb_name=kb_name,
            backend=target_backend,
            source_chars=source_chars,
        )
        baseline_records = _session_records_for_backend(
            output_root=output_root,
            kb_name=kb_name,
            backend=baseline_backend,
            source_chars=source_chars,
        )
        for key in sorted(set(target_records) & set(baseline_records)):
            matched.append((target_records[key], baseline_records[key]))
        for key in sorted(set(target_records) - set(baseline_records)):
            unmatched.append({"kb_name": kb_name, "backend": baseline_backend, "missing_match_for": list(key)})
        for key in sorted(set(baseline_records) - set(target_records)):
            unmatched.append({"kb_name": kb_name, "backend": target_backend, "missing_match_for": list(key)})

    rng.shuffle(matched)
    if limit_pairs > 0:
        matched = matched[:limit_pairs]

    package_rows = []
    key_rows = []
    for idx, (target, baseline) in enumerate(matched, start=1):
        pair_id = f"hp_{idx:06d}"
        target_is_a = rng.choice([True, False])
        system_a = target if target_is_a else baseline
        system_b = baseline if target_is_a else target
        package_rows.append(
            {
                "pair_id": pair_id,
                "rubric_version": RUBRIC_VERSION,
                "preference_values": ["A", "B", "tie"],
                "profile": target["profile"],
                "task": target["task"],
                "gaps": target["gaps"],
                "system_a": _public_system(system_a),
                "system_b": _public_system(system_b),
            }
        )
        key_rows.append(
            {
                "pair_id": pair_id,
                "kb_name": target["kb_name"],
                "profile_id": target["profile_id"],
                "entry_id": target["entry_id"],
                "session_index": target["session_index"],
                "target_backend": target_backend,
                "baseline_backend": baseline_backend,
                "system_a_backend": system_a["backend"],
                "system_b_backend": system_b["backend"],
                "system_a_transcript_path": system_a["transcript_path"],
                "system_b_transcript_path": system_b["transcript_path"],
                "system_a_evaluation_path": system_a["evaluation_path"],
                "system_b_evaluation_path": system_b["evaluation_path"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / "annotation_package.jsonl"
    key_path = output_dir / "annotation_key.json"
    template_path = output_dir / "annotation_template.csv"
    rubric_path = output_dir / "rubric.md"
    review_ui_path = output_dir / "review_ui.html"
    manifest_path = output_dir / "manifest.json"

    write_jsonl(package_path, package_rows)
    write_json(key_path, {"rubric_version": RUBRIC_VERSION, "items": key_rows})
    _write_template(template_path, [row["pair_id"] for row in package_rows])
    rubric_path.write_text(RUBRIC_MARKDOWN, encoding="utf-8")
    if REVIEW_UI_SOURCE.exists():
        review_ui_path.write_text(REVIEW_UI_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "step": "human_alignment_export_pairwise_annotations",
        "timestamp": datetime.now().isoformat(),
        "rubric_version": RUBRIC_VERSION,
        "output_root": str(output_root),
        "kb_names": kb_names,
        "target_backend": target_backend,
        "baseline_backend": baseline_backend,
        "num_pairs": len(package_rows),
        "num_unmatched_sessions": len(unmatched),
        "unmatched_sessions": unmatched,
        "limit_pairs": limit_pairs,
        "source_chars_per_page": source_chars,
        "package_path": str(package_path),
        "annotation_key_path": str(key_path),
        "annotation_template_path": str(template_path),
        "rubric_path": str(rubric_path),
        "review_ui_path": str(review_ui_path),
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export blind pairwise human annotation packages")
    parser.add_argument("--kb-names", required=True, help="Comma-separated KB names")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Pipeline output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <output_root>/human_alignment_pairwise)",
    )
    parser.add_argument("--target-backend", default="deep_tutor", help="Target backend")
    parser.add_argument("--baseline-backend", default="mock", help="Baseline backend")
    parser.add_argument("--source-chars", type=int, default=1500, help="Max source excerpt chars per page")
    parser.add_argument("--limit-pairs", type=int, default=0, help="Optional total pair cap; 0 keeps all")
    parser.add_argument("--seed", type=int, default=13, help="Randomization seed")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "human_alignment_pairwise"
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    manifest = export_annotation_package(
        output_root=output_root,
        kb_names=_parse_names(args.kb_names),
        output_dir=output_dir,
        target_backend=args.target_backend,
        baseline_backend=args.baseline_backend,
        source_chars=args.source_chars,
        limit_pairs=args.limit_pairs,
        seed=args.seed,
    )
    print(f"Annotation package: {manifest['package_path']}")
    print(f"Annotation template: {manifest['annotation_template_path']}")
    print(f"Review UI: {manifest['review_ui_path']}")
    print(f"Private key: {manifest['annotation_key_path']}")
    print(f"Pairs: {manifest['num_pairs']} | Unmatched sessions: {manifest['num_unmatched_sessions']}")


if __name__ == "__main__":
    main()
