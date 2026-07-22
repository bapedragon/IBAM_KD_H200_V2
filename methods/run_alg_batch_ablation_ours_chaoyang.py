#!/usr/bin/env python3
"""Run four ALG batch-ablation/Ours tasks in one H200 Pod."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POD_LIMIT_SECONDS = 600 * 60


@dataclass(frozen=True)
class Task:
    label: str
    method: str
    dataset: str
    batch_size: int
    protocol_name: str
    script: Path


TASKS = (
    Task(
        label="ALG-Flowers-b64",
        method="ALG",
        dataset="flowers102",
        batch_size=64,
        protocol_name="flowers102_deit_ti_alg_paper_lg_v2_trainval_test_b64",
        script=Path("methods/ALG/flowers102/train_official_split.py"),
    ),
    Task(
        label="ALG-Chaoyang-b128",
        method="ALG",
        dataset="chaoyang",
        batch_size=128,
        protocol_name="chaoyang_deit_ti_alg_paper_lg_v2_b128",
        script=Path("methods/ALG/chaoyang/train_pure_alg.py"),
    ),
    Task(
        label="ALG-Chaoyang-b64",
        method="ALG",
        dataset="chaoyang",
        batch_size=64,
        protocol_name="chaoyang_deit_ti_alg_paper_lg_v2_b64",
        script=Path("methods/ALG/chaoyang/train_pure_alg.py"),
    ),
    Task(
        label="Ours-Chaoyang-b64",
        method="Ours",
        dataset="chaoyang",
        batch_size=64,
        protocol_name="chaoyang_deit_ti_ours_cifar100_locked_b64_v1",
        script=Path("methods/Ours/chaoyang/train.py"),
    ),
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
        help="Run two full-data epochs per task and estimate the 300-epoch total.",
    )
    mode.add_argument(
        "--full-run",
        action="store_true",
        help="Run all four tasks for 300 epochs each.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument(
        "--chaoyang-data-dir",
        type=Path,
        default=Path("/app/data/chaoyang"),
    )
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=REPOSITORY_ROOT / "teachers/checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to /tmp for timing and singular /app/output for full runs.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def run_name(task: Task, timing_run: bool) -> str:
    mode = "timing_2ep" if timing_run else "300ep"
    return f"{task.protocol_name}_{mode}_seed1"


def build_command(
    task: Task,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[list[str], Path]:
    task_output = output_root / task.method / task.dataset / f"batch{task.batch_size}"
    name = run_name(task, args.timing_run)
    task_data_dir = (
        args.chaoyang_data_dir if task.dataset == "chaoyang" else args.data_dir
    )
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / task.script),
        "--data-dir",
        str(task_data_dir),
        "--teacher-root",
        str(args.teacher_root),
        "--output-dir",
        str(task_output),
        "--run-name",
        name,
        "--protocol-name",
        task.protocol_name,
        "--batch-size",
        str(task.batch_size),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.timing_run:
        command.append("--timing-run")
    return command, task_output / name


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    default_root = (
        Path("/tmp/alg_batch_ablation_ours_chaoyang_timing")
        if args.timing_run
        else Path("/app/output/alg_batch_ablation_ours_chaoyang_300ep_seed1")
    )
    output_root = (args.output_dir or default_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "sequence_status.json"
    summary_path = output_root / "sequence_summary.json"
    records: list[dict[str, Any]] = []
    sequence_start = time.time()

    log("=" * 80)
    log("PURE ALG BATCH ABLATION + CIFAR-LOCKED OURS CHAOYANG")
    log("=" * 80)
    log(f"[MODE] timing_run={args.timing_run} full_run={args.full_run}")
    log("[SEQUENCE] ALG Flowers-b64 -> ALG Chaoyang-b128 -> ALG Chaoyang-b64")
    log("[SEQUENCE] -> Ours Chaoyang-b64")
    log("[COMMON] epochs=300 eval_batch=200 AdamW lr=5e-4 min_lr=5e-6")
    log("[COMMON] wd=0.05 optimizer_warmup=20 seed=1 FP32 teacher=32 student=224")
    log("[PURE_ALG] LG feature loss beta=2.5 paper_derivative paper_ge warmup=0")
    log("[OURS] researcher_sync_v1 larger_grid combined_loss controller_warmup=20")
    log(f"[PATH] data_dir={args.data_dir.resolve()}")
    log(f"[PATH] chaoyang_data_dir={args.chaoyang_data_dir.resolve()}")
    log(f"[PATH] teacher_root={args.teacher_root.resolve()}")
    log(f"[PATH] output_root={output_root}")
    log("[OUTPUT] Four independent run directories; no checkpoint can overwrite another.")

    try:
        for order, task in enumerate(TASKS, start=1):
            command, run_dir = build_command(task, args, output_root)
            record: dict[str, Any] = {
                "order": order,
                "label": task.label,
                "method": task.method,
                "dataset": task.dataset,
                "batch_size": task.batch_size,
                "protocol_name": task.protocol_name,
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
            log(f"[SEQUENCE][{order}/{len(TASKS)}] START {task.label}")
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
                    "latest_top1": task_summary.get("latest_top1"),
                    "best_epoch": task_summary.get("best_epoch"),
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
                f"[SEQUENCE][{order}/{len(TASKS)}] DONE {task.label} "
                f"best_top1={record['best_top1']}"
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
        "protocol": "pure_alg_batch_ablation_plus_cifar_locked_ours_v1",
        "planned_epochs_each": 300,
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
    for record in records:
        log(
            f"[FINAL_BEST][{record['label']}] "
            f"best_top1={float(record['best_top1']):.2f}%"
        )
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
    log("[DONE] All four ALG/Ours tasks completed successfully.")


def cli_main() -> None:
    try:
        main()
    except Exception as error:
        log("=" * 80)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Four-task ALG/Ours sequence did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
