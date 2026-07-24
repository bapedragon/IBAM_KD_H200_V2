#!/usr/bin/env python3
"""Train one CUB-200 teacher, then official LG, paper ALG, and Ours."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POD_LIMIT_SECONDS = 600 * 60
TEACHER_RUN_NAME = "teacher_cub200_resnet56_32_scratch_300ep_seed1"
STUDENT_RUN_NAMES = {
    "LG": "lg_cub200_deit_ti_300ep_seed1",
    "ALG": "alg_cub200_deit_ti_300ep_seed1",
    "Ours": "ours_cub200_deit_ti_300ep_seed1",
}
STUDENT_SCRIPTS = {
    "LG": "methods/LG/cub200/train.py",
    "ALG": "methods/ALG/cub200/train.py",
    "Ours": "methods/Ours/cub200/train.py",
}


def log(message: str = "") -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--timing-run", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("./data/cub200"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/output/cub200_lg_alg_ours_300ep_seed1"),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_tracked_task(
    *,
    name: str,
    command: list[str],
    summary_path: Path,
    status: dict[str, Any],
    status_path: Path,
) -> dict[str, Any]:
    log(f"[COMMAND][{name}] {' '.join(command)}")
    try:
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as error:
        status.update(
            {
                "status": "failed",
                "failed_task": name,
                "error": f"{type(error).__name__}: {error}",
                "finished_at_unix": time.time(),
            }
        )
        status["tasks"].append({"name": name, "status": "failed"})
        atomic_json_save(status, status_path)
        raise
    status["tasks"].append(
        {"name": name, "status": "complete", "summary": str(summary_path)}
    )
    status["completed_tasks"] += 1
    atomic_json_save(status, status_path)
    return summary


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    timing = bool(args.timing_run)
    mode = "timing" if timing else "full"
    suffix = "_timing_2ep" if timing else ""
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "sequence_status.json"
    teacher_name = TEACHER_RUN_NAME + suffix
    teacher_output = output_root / "teacher"
    teacher_root = teacher_output / teacher_name
    status: dict[str, Any] = {
        "status": "running",
        "mode": mode,
        "completed_tasks": 0,
        "total_tasks": 4,
        "tasks": [],
        "started_at_unix": time.time(),
        "protocols": {
            "LG": "official LG code mechanics, static guidance",
            "ALG": "official LG base plus ALG paper controller",
            "Ours": "unchanged repository Ours protocol",
        },
    }
    atomic_json_save(status, status_path)

    log("=" * 80)
    log("CUB-200: SHARED SCRATCH TEACHER -> OFFICIAL LG -> PAPER ALG -> OURS")
    log("=" * 80)
    log(f"[MODE] {mode}")
    log("[PROTOCOL_LOCK] LG=official ALG=paper_on_official_LG Ours=unchanged")
    log("[FAIRNESS] all students consume the exact same teacher manifest/checkpoint")
    log(f"[PATH] data_dir={args.data_dir.resolve()}")
    log(f"[PATH] output_root={output_root}")

    teacher_command = [
        sys.executable,
        "teachers/train_teacher_cub200.py",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(teacher_output),
        "--run-name",
        teacher_name,
        "--num-workers",
        str(args.num_workers),
    ]
    if timing:
        teacher_command.append("--timing-run")
    if args.no_download:
        teacher_command.append("--no-download")
    summaries: dict[str, dict[str, Any]] = {}
    summaries["teacher"] = run_tracked_task(
        name="teacher",
        command=teacher_command,
        summary_path=teacher_root / "summary.json",
        status=status,
        status_path=status_path,
    )

    for method in ("LG", "ALG", "Ours"):
        run_name = STUDENT_RUN_NAMES[method] + suffix
        method_output = output_root / method / "cub200"
        command = [
            sys.executable,
            STUDENT_SCRIPTS[method],
            "--data-dir",
            str(args.data_dir),
            "--teacher-root",
            str(teacher_root),
            "--output-dir",
            str(method_output),
            "--run-name",
            run_name,
            "--num-workers",
            str(args.num_workers),
        ]
        if timing:
            command.extend(("--timing-run", "--allow-teacher-runtime-gap"))
        summaries[method] = run_tracked_task(
            name=method,
            command=command,
            summary_path=method_output / run_name / "summary.json",
            status=status,
            status_path=status_path,
        )

    estimated_seconds = sum(
        float(summary["estimated_planned_seconds"])
        for summary in summaries.values()
    )
    margin = POD_LIMIT_SECONDS - estimated_seconds
    limit_status = "PASS" if margin > 0 else "FAIL"
    status.update(
        {
            "status": "complete",
            "estimated_planned_seconds": estimated_seconds,
            "pod_limit_seconds": POD_LIMIT_SECONDS,
            "pod_limit_status": limit_status,
            "finished_at_unix": time.time(),
        }
    )
    atomic_json_save(status, status_path)
    log("=" * 80)
    log("[SEQUENCE_DONE] completed_tasks=4/4")
    log(
        f"[POD_LIMIT_CHECK] status={limit_status} "
        f"estimated_minutes={estimated_seconds / 60:.1f} "
        f"limit_minutes={POD_LIMIT_SECONDS / 60:.0f} "
        f"{'margin' if margin >= 0 else 'over_by'}_minutes={abs(margin) / 60:.1f}"
    )
    if timing and limit_status == "FAIL":
        log(
            "[NEXT_ACTION] Split the full run into teacher/LG/ALG/Ours Issues "
            "that share a persisted full-teacher directory."
        )
    log(f"[FINAL_RESULT] sequence_status={status_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"[FATAL] {type(error).__name__}: {error}")
        raise
