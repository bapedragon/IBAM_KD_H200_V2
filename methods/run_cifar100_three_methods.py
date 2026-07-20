#!/usr/bin/env python3
"""Run CIFAR-100 DeiT-Ti KD, CRD, and ReviewKD sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    ("KD", Path("methods/KD/cifar100/train.py"), "kd"),
    ("CRD", Path("methods/CRD/cifar100/train.py"), "crd"),
    ("ReviewKD", Path("methods/ReviewKD/cifar100/train.py"), "reviewkd"),
)


def log(message: str = "") -> None:
    print(message, flush=True)


def format_duration(seconds: float) -> str:
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--timing-run",
        action="store_true",
        help="Run two full-data epochs per method using the 300-epoch schedule.",
    )
    mode.add_argument(
        "--full-run",
        action="store_true",
        help="Run all three methods for the full 300 epochs.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=REPOSITORY_ROOT / "teachers/checkpoints",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_name(prefix: str, timing_run: bool, seed: int) -> str:
    suffix = "timing_2ep" if timing_run else "full_300ep"
    return f"{prefix}_cifar100_deit_ti_{suffix}_seed{seed}_v2"


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "three_method_status.json"
    sequence_start = time.time()
    records: list[dict[str, Any]] = []

    log("=" * 80)
    log("CIFAR-100 DEIT-TI SEQUENTIAL DISTILLATION: KD -> CRD -> REVIEWKD")
    log("=" * 80)
    log(f"[MODE] timing_run={args.timing_run} full_run={args.full_run}")
    log(f"[PATH] repository={REPOSITORY_ROOT}")
    log(f"[PATH] data_dir={args.data_dir.resolve()}")
    log(f"[PATH] teacher_root={args.teacher_root.resolve()}")
    log(f"[PATH] output_root={output_root}")
    log(
        "[PROTOCOL] dataset=CIFAR-100 student=DeiT-Ti student_input=224 "
        f"teacher=ResNet56 teacher_input=32 epochs=300 "
        f"batch={args.batch_size} seed={args.seed}"
    )
    log(
        "[OUTPUT] Each method writes an independent run directory; "
        "checkpoints cannot overwrite another method."
    )

    verify_command = [
        sys.executable,
        str(REPOSITORY_ROOT / "teachers/verify_checkpoints.py"),
        "--dataset",
        "cifar100",
        "--checkpoint-root",
        str(args.teacher_root),
    ]
    log(f"[PRECHECK] command={' '.join(verify_command)}")
    subprocess.run(verify_command, cwd=REPOSITORY_ROOT, check=True)

    try:
        for index, (method, relative_script, prefix) in enumerate(METHODS, start=1):
            name = run_name(prefix, args.timing_run, args.seed)
            method_dir = output_root / name
            command = [
                sys.executable,
                str(REPOSITORY_ROOT / relative_script),
                "--protocol-name",
                "cifar100_deit_ti_common_kd_v2",
                "--data-dir",
                str(args.data_dir),
                "--teacher-root",
                str(args.teacher_root),
                "--output-dir",
                str(output_root),
                "--run-name",
                name,
                "--student-epochs",
                "300",
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--image-size",
                "224",
                "--seed",
                str(args.seed),
            ]
            if args.timing_run:
                command.append("--timing-run")

            record: dict[str, Any] = {
                "order": index,
                "method": method,
                "status": "running",
                "run_dir": str(method_dir),
                "command": command,
            }
            records.append(record)
            atomic_json(
                status_path,
                {
                    "status": "running",
                    "mode": "timing" if args.timing_run else "full",
                    "records": records,
                },
            )
            log("=" * 80)
            log(f"[SEQUENCE][{index}/3] START method={method}")
            log(f"[SEQUENCE][{index}/3] command={' '.join(command)}")
            method_start = time.time()
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
            record["status"] = "complete"
            record["elapsed_seconds"] = time.time() - method_start
            summary_path = method_dir / "summary.json"
            record["summary"] = str(summary_path)
            if not summary_path.is_file():
                raise FileNotFoundError(
                    f"{method} completed without summary: {summary_path}"
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            record["best_top1"] = summary.get("best_top1")
            record["estimated_planned_human"] = summary.get(
                "estimated_planned_human"
            )
            record["estimated_planned_seconds"] = summary.get(
                "estimated_planned_seconds"
            )
            atomic_json(
                status_path,
                {
                    "status": "running" if index < len(METHODS) else "complete",
                    "mode": "timing" if args.timing_run else "full",
                    "records": records,
                },
            )
            log(
                f"[SEQUENCE][{index}/3] DONE method={method} "
                f"best_top1={record['best_top1']} run_dir={method_dir}"
            )
    except Exception as error:
        if records and records[-1]["status"] == "running":
            records[-1]["status"] = "failed"
            records[-1]["error"] = f"{type(error).__name__}: {error}"
        atomic_json(
            status_path,
            {
                "status": "failed",
                "mode": "timing" if args.timing_run else "full",
                "records": records,
            },
        )
        raise

    elapsed = time.time() - sequence_start
    estimated_full_seconds = sum(
        float(record.get("estimated_planned_seconds") or 0.0)
        for record in records
    )
    final_summary = {
        "status": "complete",
        "mode": "timing" if args.timing_run else "full",
        "dataset": "cifar100",
        "student": "deit_ti",
        "teacher_input": 32,
        "student_input": 224,
        "elapsed_seconds": elapsed,
        "elapsed_human": format_duration(elapsed),
        "estimated_three_method_full_seconds": estimated_full_seconds,
        "estimated_three_method_full_human": format_duration(
            estimated_full_seconds
        ),
        "records": records,
    }
    final_path = output_root / "three_method_summary.json"
    atomic_json(final_path, final_summary)
    log("=" * 80)
    log("[FINAL_RESULT] completed_methods=KD,CRD,ReviewKD")
    for record in records:
        log(
            f"[FINAL_RESULT] method={record['method']} "
            f"best_top1={record['best_top1']} run_dir={record['run_dir']}"
        )
    log(f"[FINAL_RESULT] aggregate_summary={final_path}")
    log(
        f"[TIMING] estimated_three_method_full="
        f"{format_duration(estimated_full_seconds)} "
        f"({estimated_full_seconds:.1f}s)"
    )
    log("[DONE] All three methods completed successfully; resources may be released.")


def cli_main() -> None:
    try:
        main()
    except Exception as error:
        log("=" * 80)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Three-method sequence did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
