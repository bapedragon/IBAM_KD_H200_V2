#!/usr/bin/env python3
"""Run CIFAR-100 DeiT-Ti Ours, CRD, and MGD sequentially."""

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
    ("Ours", Path("methods/Ours/cifar100/train.py"), "ours"),
    ("CRD", Path("methods/CRD/cifar100/train.py"), "crd"),
    ("MGD", Path("methods/MGD/cifar100/train.py"), "mgd"),
)
MEASURED_FULL_SECONDS = {
    "Ours": 4 * 3600 + 8 * 60 + 37,
    "CRD": 3 * 3600 + 15 * 60 + 4,
    "MGD": 3 * 3600 + 3 * 60 + 26,
}
POD_LIMIT_SECONDS = 600 * 60


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
        help="Run the selected methods for the full 300 epochs.",
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
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[method for method, _, _ in METHODS],
        default=[method for method, _, _ in METHODS],
        help=(
            "Subset to execute in canonical Ours -> CRD -> MGD order. "
            "Defaults to all three. "
            "Use this to split a full run across the H200 runtime limit."
        ),
    )
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

    requested = set(args.methods)
    selected_methods = tuple(item for item in METHODS if item[0] in requested)
    selected_names = [item[0] for item in selected_methods]
    if len(selected_names) != len(args.methods):
        raise ValueError("--methods must not contain duplicate method names")
    measured_selected_seconds = sum(
        MEASURED_FULL_SECONDS[name] for name in selected_names
    )
    if args.full_run and measured_selected_seconds >= POD_LIMIT_SECONDS:
        raise RuntimeError(
            "Selected full-run methods exceed the 600-minute Pod limit: "
            f"methods={','.join(selected_names)} "
            f"measured={format_duration(measured_selected_seconds)}. "
            "Run Ours alone and CRD+MGD in separate Issues."
        )

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "ours_crd_mgd_status.json"
    sequence_start = time.time()
    records: list[dict[str, Any]] = []

    log("=" * 80)
    log(
        "CIFAR-100 DEIT-TI SEQUENTIAL DISTILLATION: "
        + " -> ".join(selected_names)
    )
    log("=" * 80)
    log(f"[MODE] timing_run={args.timing_run} full_run={args.full_run}")
    log(f"[METHODS] selected={','.join(selected_names)}")
    log(
        f"[RUNTIME_PLAN] measured_full={format_duration(measured_selected_seconds)} "
        f"pod_limit={format_duration(POD_LIMIT_SECONDS)} "
        f"margin={format_duration(POD_LIMIT_SECONDS - measured_selected_seconds)}"
    )
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
        total_methods = len(selected_methods)
        for index, (method, relative_script, prefix) in enumerate(
            selected_methods, start=1
        ):
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
                    "selected_methods": selected_names,
                    "records": records,
                },
            )
            log("=" * 80)
            log(f"[SEQUENCE][{index}/{total_methods}] START method={method}")
            log(f"[SEQUENCE][{index}/{total_methods}] command={' '.join(command)}")
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
                    "status": "running" if index < total_methods else "complete",
                    "mode": "timing" if args.timing_run else "full",
                    "selected_methods": selected_names,
                    "records": records,
                },
            )
            log(
                f"[SEQUENCE][{index}/{total_methods}] DONE method={method} "
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
                "selected_methods": selected_names,
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
        "selected_methods": selected_names,
        "elapsed_seconds": elapsed,
        "elapsed_human": format_duration(elapsed),
        "estimated_selected_methods_full_seconds": estimated_full_seconds,
        "estimated_selected_methods_full_human": format_duration(
            estimated_full_seconds
        ),
        "records": records,
    }
    if len(selected_names) == len(METHODS):
        final_summary["estimated_three_method_full_seconds"] = (
            estimated_full_seconds
        )
        final_summary["estimated_three_method_full_human"] = format_duration(
            estimated_full_seconds
        )
    final_path = output_root / "ours_crd_mgd_summary.json"
    atomic_json(final_path, final_summary)
    log("=" * 80)
    log(f"[FINAL_RESULT] completed_methods={','.join(selected_names)}")
    for record in records:
        log(
            f"[FINAL_RESULT] method={record['method']} "
            f"best_top1={record['best_top1']} run_dir={record['run_dir']}"
        )
    log(f"[FINAL_RESULT] aggregate_summary={final_path}")
    log(
        f"[TIMING] estimated_selected_methods_full="
        f"{format_duration(estimated_full_seconds)} "
        f"({estimated_full_seconds:.1f}s)"
    )
    if len(selected_names) == len(METHODS):
        log(
            "[DONE] Ours, CRD, and MGD completed successfully; "
            "resources may be released."
        )
    else:
        log(
            "[DONE] Selected methods completed successfully: "
            f"{','.join(selected_names)}; resources may be released."
        )


def cli_main() -> None:
    try:
        main()
    except Exception as error:
        log("=" * 80)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Method sequence did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
