#!/usr/bin/env python3
"""Run the CUB-200 scratch teacher followed by Ours student training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
POD_LIMIT_SECONDS = 600 * 60
TEACHER_RUN_NAME = "teacher_cub200_resnet56_32_scratch_300ep_seed1"
STUDENT_RUN_NAME = "ours_cub200_deit_ti_300ep_seed1"


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
        default=Path("/app/output/cub200_ours_pipeline_300ep_seed1"),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_command(command: list[str]) -> None:
    log(f"[COMMAND] {' '.join(command)}")
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def run_tracked_task(
    *,
    name: str,
    command: list[str],
    summary_path: Path,
    status: dict[str, Any],
    status_path: Path,
) -> dict[str, Any]:
    try:
        run_command(command)
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
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "sequence_status.json"
    timing = args.timing_run
    mode = "timing" if timing else "full"
    suffix = "_timing_2ep" if timing else ""
    teacher_name = TEACHER_RUN_NAME + suffix
    student_name = STUDENT_RUN_NAME + suffix
    teacher_output = output_root / "teacher"
    teacher_root = teacher_output / teacher_name
    student_output = output_root / "Ours" / "cub200"

    status: dict[str, Any] = {
        "status": "running",
        "mode": mode,
        "completed_tasks": 0,
        "total_tasks": 2,
        "tasks": [],
        "started_at_unix": time.time(),
    }
    atomic_json_save(status, status_path)
    log("=" * 80)
    log("CUB-200-2011 SCRATCH TEACHER -> OURS PIPELINE")
    log("=" * 80)
    log(f"[MODE] mode={mode}")
    log(f"[PATH] data_dir={args.data_dir.resolve()}")
    log(f"[PATH] output_root={output_root}")

    teacher_command = [
        sys.executable,
        "methods/Ours/cub200/train_teacher.py",
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
    teacher_summary_path = teacher_root / "summary.json"
    teacher_summary = run_tracked_task(
        name="teacher",
        command=teacher_command,
        summary_path=teacher_summary_path,
        status=status,
        status_path=status_path,
    )

    student_command = [
        sys.executable,
        "methods/Ours/cub200/train.py",
        "--data-dir",
        str(args.data_dir),
        "--teacher-root",
        str(teacher_root),
        "--output-dir",
        str(student_output),
        "--run-name",
        student_name,
        "--num-workers",
        str(args.num_workers),
    ]
    if timing:
        student_command.extend(("--timing-run", "--allow-teacher-runtime-gap"))
    student_summary_path = student_output / student_name / "summary.json"
    student_summary = run_tracked_task(
        name="Ours",
        command=student_command,
        summary_path=student_summary_path,
        status=status,
        status_path=status_path,
    )
    estimated_seconds = float(
        teacher_summary["estimated_planned_seconds"]
    ) + float(student_summary["estimated_planned_seconds"])
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
    log("[SEQUENCE_DONE] completed_tasks=2/2")
    log(
        f"[POD_LIMIT_CHECK] status={limit_status} "
        f"estimated_minutes={estimated_seconds / 60:.1f} "
        f"limit_minutes={POD_LIMIT_SECONDS / 60:.0f} "
        f"{'margin' if margin >= 0 else 'over_by'}_minutes={abs(margin) / 60:.1f}"
    )
    if timing and limit_status == "FAIL":
        log(
            "[NEXT_ACTION] Submit teacher and Ours as separate full-run Issues; "
            "do not submit the combined full command."
        )
    log(f"[FINAL_RESULT] sequence_status={status_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"[FATAL] {type(error).__name__}: {error}")
        raise
