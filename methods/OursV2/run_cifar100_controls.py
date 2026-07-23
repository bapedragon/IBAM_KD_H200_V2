#!/usr/bin/env python3
"""Run Ours V2 grid-permutation and token-space controls sequentially."""

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


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
POD_LIMIT_SECONDS = 600 * 60


@dataclass(frozen=True)
class Task:
    label: str
    variant: str
    protocol_name: str
    script: Path


TASKS = (
    Task(
        label="OursV2-Grid-Permutation",
        variant="grid_permutation_v1",
        protocol_name="cifar100_deit_ti_ours_v2_grid_permutation_v1",
        script=Path("methods/OursV2/grid_permutation/train_cifar100.py"),
    ),
    Task(
        label="OursV2-Token-Space",
        variant="token_space_v1",
        protocol_name="cifar100_deit_ti_ours_v2_token_space_v1",
        script=Path("methods/OursV2/token_space/train_cifar100.py"),
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
        help="Run two full-data epochs per control and estimate 300 epochs.",
    )
    mode.add_argument(
        "--full-run",
        action="store_true",
        help="Run both controls for 300 epochs each.",
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
            "Defaults to /tmp for timing and singular /app/output for full runs. "
            "Each control receives an independent subdirectory."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def task_run_name(task: Task, timing_run: bool) -> str:
    mode = "timing_2ep" if timing_run else "300ep"
    return f"{task.protocol_name}_{mode}_seed1"


def build_command(
    task: Task,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[list[str], Path]:
    task_output = output_root / "OursV2" / "cifar100" / task.variant
    run_name = task_run_name(task, args.timing_run)
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / task.script),
        "--data-dir",
        str(args.data_dir),
        "--teacher-root",
        str(args.teacher_root),
        "--output-dir",
        str(task_output),
        "--run-name",
        run_name,
        "--protocol-name",
        task.protocol_name,
        "--num-workers",
        str(args.num_workers),
    ]
    if args.timing_run:
        command.append("--timing-run")
    return command, task_output / run_name


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    default_root = (
        Path("/tmp/ours_v2_cifar100_controls_timing")
        if args.timing_run
        else Path("/app/output/ours_v2_cifar100_controls_300ep_seed1")
    )
    output_root = (args.output_dir or default_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "sequence_status.json"
    summary_path = output_root / "sequence_summary.json"
    records: list[dict[str, Any]] = []
    sequence_start = time.time()

    log("=" * 80)
    log("OURS V2 CIFAR-100 CONTROLS: GRID PERMUTATION -> TOKEN SPACE")
    log("=" * 80)
    log(f"[MODE] timing_run={args.timing_run} full_run={args.full_run}")
    log("[SEQUENCE] grid_permutation_v1 -> token_space_v1")
    log("[PROTOCOL_LOCK] epochs=300 train/eval_batch=64/200 AdamW")
    log("[PROTOCOL_LOCK] lr=5e-4 min_lr=5e-6 wd=0.05 warmup=20")
    log("[PROTOCOL_LOCK] seed=1 FP32 teacher=32 student=224 grids=32/16/14")
    log(
        "[GRID_PERMUTATION] joint_same_order_KV=True "
        "relative_2d_bias=True mse_targets=unpermuted"
    )
    log(
        "[TOKEN_SPACE] qkv=Linear_on_flattened_BNC "
        "relative_2d_bias=False"
    )
    log(f"[PATH] data_dir={args.data_dir.resolve()}")
    log(f"[PATH] teacher_root={args.teacher_root.resolve()}")
    log(f"[PATH] output_root={output_root}")
    log("[OUTPUT] Two independent run directories; no V1 artifact is touched.")

    try:
        for order, task in enumerate(TASKS, start=1):
            command, run_dir = build_command(task, args, output_root)
            record: dict[str, Any] = {
                "order": order,
                "label": task.label,
                "variant": task.variant,
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
            log(
                f"[SEQUENCE][{order}/{len(TASKS)}] START "
                f"variant={task.variant}"
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
            recorded_variant = task_summary.get("ours_v2", {}).get("variant")
            if task_summary.get("method") != "OursV2":
                raise RuntimeError(
                    "Task summary method mismatch: "
                    f"expected=OursV2 got={task_summary.get('method')}"
                )
            if recorded_variant != task.variant:
                raise RuntimeError(
                    "Task summary variant mismatch: "
                    f"expected={task.variant} got={recorded_variant}"
                )

            record.update(
                {
                    "status": "complete",
                    "summary": str(task_summary_path),
                    "best_top1": task_summary.get("best_top1"),
                    "latest_top1": task_summary.get("latest_top1"),
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
                f"[SEQUENCE][{order}/{len(TASKS)}] DONE "
                f"variant={task.variant} "
                f"estimated_300ep={record['estimated_planned_human']}"
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

    elapsed_seconds = time.time() - sequence_start
    estimated_full_seconds = sum(
        float(record.get("estimated_planned_seconds") or 0.0)
        for record in records
    )
    pod_limit_passed = estimated_full_seconds < POD_LIMIT_SECONDS
    pod_limit_delta_seconds = abs(POD_LIMIT_SECONDS - estimated_full_seconds)
    payload = {
        "status": "complete",
        "mode": "timing" if args.timing_run else "full",
        "protocol": "ours_v2_cifar100_controls_v1",
        "planned_epochs_each": 300,
        "task_order": [task.variant for task in TASKS],
        "elapsed_seconds": elapsed_seconds,
        "elapsed_human": format_duration(elapsed_seconds),
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
            f"[FINAL_ESTIMATE][{record['variant']}] "
            f"avg_epoch={float(record['avg_epoch_seconds']):.2f}s "
            f"estimated_300ep={record['estimated_planned_human']}"
        )
    log(
        f"[SEQUENCE_DONE] completed_tasks={len(records)}/{len(TASKS)} "
        f"estimated_combined_300ep={format_duration(estimated_full_seconds)}"
    )
    log(
        f"[POD_LIMIT_CHECK] status={'PASS' if pod_limit_passed else 'FAIL'} "
        f"limit=10h 00m 00s estimated={format_duration(estimated_full_seconds)} "
        f"{'headroom' if pod_limit_passed else 'over_by'}="
        f"{format_duration(pod_limit_delta_seconds)}"
    )
    log(f"[FINAL_RESULT] sequence_summary={summary_path}")
    log("[DONE] Ours V2 CIFAR-100 control sequence completed successfully.")


def cli_main() -> None:
    try:
        main()
    except Exception as error:
        log("=" * 80)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Ours V2 CIFAR-100 control sequence did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
