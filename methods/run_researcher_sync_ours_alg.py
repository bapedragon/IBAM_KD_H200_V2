#!/usr/bin/env python3
"""Run researcher-sync Ours CIFAR/Flowers and ALG Flowers sequentially."""

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
TASKS = (
    ("Ours", "cifar100", Path("methods/Ours/cifar100/train.py")),
    ("Ours", "flowers102", Path("methods/Ours/flowers102/train.py")),
    ("ALG", "flowers102", Path("methods/ALG/flowers102/train.py")),
)

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
        help="Run two full-data epochs for each 300-epoch task.",
    )
    mode.add_argument(
        "--full-run",
        action="store_true",
        help="Run all three tasks for 300 epochs.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=REPOSITORY_ROOT / "teachers/checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Defaults to /tmp for timing and /app/output for full runs. "
            "Every task receives a unique subdirectory."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def task_run_name(method: str, dataset: str, timing_run: bool) -> str:
    mode = "timing_2ep" if timing_run else "300ep"
    return f"{method.lower()}_{dataset}_deit_ti_researcher_sync_{mode}_seed1"


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    default_root = (
        Path("/tmp/researcher_sync_ours_alg_timing")
        if args.timing_run
        else Path("/app/output/researcher_sync_ours_alg_300ep_seed1")
    )
    output_root = (args.output_dir or default_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "sequence_status.json"
    summary_path = output_root / "sequence_summary.json"
    records: list[dict[str, Any]] = []
    sequence_start = time.time()

    log("=" * 80)
    log("RESEARCHER-SYNC SEQUENCE: OURS CIFAR -> OURS FLOWERS -> ALG FLOWERS")
    log("=" * 80)
    log(f"[MODE] timing_run={args.timing_run} full_run={args.full_run}")
    log("[PROTOCOL_LOCK] epochs=300 train/eval_batch=64/200 AdamW")
    log("[PROTOCOL_LOCK] lr=5e-4 min_lr=5e-6 wd=0.05 warmup=20")
    log("[PROTOCOL_LOCK] seed=1 FP32 public_LG_augmentation researcher_controller")
    log("[PROTOCOL_LOCK] teacher=32px student=224px grid=larger direct_eval")
    log(f"[PATH] data_dir={args.data_dir.resolve()}")
    log(f"[PATH] teacher_root={args.teacher_root.resolve()}")
    log(f"[PATH] output_root={output_root}")
    log(
        "[OUTPUT] Independent method/dataset/run directories prevent all "
        "cross-task and historical-result overwrites."
    )

    try:
        for order, (method, dataset, relative_script) in enumerate(TASKS, start=1):
            run_name = task_run_name(method, dataset, args.timing_run)
            task_output = output_root / method / dataset
            run_dir = task_output / run_name
            command = [
                sys.executable,
                str(REPOSITORY_ROOT / relative_script),
                "--data-dir",
                str(args.data_dir),
                "--teacher-root",
                str(args.teacher_root),
                "--output-dir",
                str(task_output),
                "--run-name",
                run_name,
                "--num-workers",
                str(args.num_workers),
            ]
            if args.timing_run:
                command.append("--timing-run")

            record: dict[str, Any] = {
                "order": order,
                "method": method,
                "dataset": dataset,
                "status": "running",
                "run_dir": str(run_dir),
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
            log(
                f"[SEQUENCE][{order}/{len(TASKS)}] START "
                f"method={method} dataset={dataset}"
            )
            log(f"[SEQUENCE][{order}/{len(TASKS)}] command={' '.join(command)}")
            task_start = time.time()
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
            record["elapsed_seconds"] = time.time() - task_start
            task_summary_path = run_dir / "summary.json"
            if not task_summary_path.is_file():
                raise FileNotFoundError(
                    f"Completed task has no summary: {task_summary_path}"
                )
            task_summary = json.loads(task_summary_path.read_text(encoding="utf-8"))
            record.update(
                {
                    "status": "complete",
                    "summary": str(task_summary_path),
                    "best_top1": task_summary.get("best_top1"),
                    "avg_epoch_seconds": task_summary.get("avg_epoch_seconds"),
                    "estimated_planned_seconds": task_summary.get(
                        "estimated_planned_seconds"
                    ),
                    "estimated_planned_human": task_summary.get(
                        "estimated_planned_human"
                    ),
                }
            )
            atomic_json(
                status_path,
                {
                    "status": "complete" if order == len(TASKS) else "running",
                    "mode": "timing" if args.timing_run else "full",
                    "records": records,
                },
            )
            log(
                f"[SEQUENCE][{order}/{len(TASKS)}] DONE method={method} "
                f"dataset={dataset} best_top1={record['best_top1']}"
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
    pod_limit_passed = estimated_full_seconds < POD_LIMIT_SECONDS
    pod_limit_delta_seconds = abs(POD_LIMIT_SECONDS - estimated_full_seconds)
    payload = {
        "status": "complete",
        "mode": "timing" if args.timing_run else "full",
        "protocol": "researcher_sync_v1",
        "planned_epochs_each": 300,
        "tasks": [f"{method}:{dataset}" for method, dataset, _ in TASKS],
        "elapsed_seconds": elapsed,
        "elapsed_human": format_duration(elapsed),
        "estimated_full_seconds": estimated_full_seconds,
        "estimated_full_human": format_duration(estimated_full_seconds),
        "pod_limit_seconds": POD_LIMIT_SECONDS,
        "pod_limit_passed": pod_limit_passed,
        "pod_limit_delta_seconds": pod_limit_delta_seconds,
        "pod_limit_delta_human": format_duration(pod_limit_delta_seconds),
        "records": records,
    }
    atomic_json(summary_path, payload)
    log("=" * 80)
    log(
        f"[FINAL_RESULT] completed_tasks={len(records)}/{len(TASKS)} "
        f"estimated_full={format_duration(estimated_full_seconds)}"
    )
    log(
        f"[POD_LIMIT_CHECK] status={'PASS' if pod_limit_passed else 'FAIL'} "
        f"limit=10h 00m 00s estimated={format_duration(estimated_full_seconds)} "
        f"{'headroom' if pod_limit_passed else 'over_by'}="
        f"{format_duration(pod_limit_delta_seconds)}"
    )
    log(f"[FINAL_RESULT] summary={summary_path}")
    log("[DONE] All researcher-sync tasks completed successfully.")


def cli_main() -> None:
    try:
        main()
    except Exception as error:
        log("=" * 80)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Researcher-sync sequence did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
