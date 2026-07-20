#!/usr/bin/env python3
"""Run Chaoyang five methods, Flowers five methods, then one CIFAR-100 method."""

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
CIFAR_METHOD_ESTIMATES = {
    "KD": 3 * 3600 + 6 * 60 + 28,
    "CRD": 3 * 3600 + 16 * 60 + 42,
    "ReviewKD": 3 * 3600 + 18 * 60 + 27,
}
FLOWERS_FIVE_METHOD_ESTIMATE = 3 * 3600 + 20 * 60 + 1
CHAOYANG_FIVE_METHOD_ESTIMATE = 1 * 3600 + 19 * 60 + 30
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
    parser.add_argument(
        "--cifar-method",
        choices=tuple(CIFAR_METHOD_ESTIMATES),
        default="KD",
        help="One measured CIFAR-100 method appended after both five-method runs.",
    )
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=REPOSITORY_ROOT / "teachers/checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/output/combined_flowers_chaoyang_cifar100_v2"),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_tasks(args: argparse.Namespace, output_root: Path) -> list[dict[str, Any]]:
    """Return the locked short-first execution plan."""

    common = [
        "--full-run",
        "--teacher-root",
        str(args.teacher_root),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
    ]
    return [
        {
            "dataset": "chaoyang",
            "methods": ["KD", "CRD", "ReviewKD", "MGD", "OFA"],
            "estimated_seconds": CHAOYANG_FIVE_METHOD_ESTIMATE,
            "output_dir": output_root / "chaoyang",
            "arguments": ["--dataset", "chaoyang", *common],
        },
        {
            "dataset": "flowers102",
            "methods": ["KD", "CRD", "ReviewKD", "MGD", "OFA"],
            "estimated_seconds": FLOWERS_FIVE_METHOD_ESTIMATE,
            "output_dir": output_root / "flowers102",
            "arguments": ["--dataset", "flowers102", *common],
        },
        {
            "dataset": "cifar100",
            "methods": [args.cifar_method],
            "estimated_seconds": CIFAR_METHOD_ESTIMATES[args.cifar_method],
            "output_dir": output_root / f"cifar100_{args.cifar_method.lower()}",
            "arguments": [
                "--dataset",
                "cifar100",
                *common,
                "--methods",
                args.cifar_method,
            ],
        },
    ]


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args, output_root)
    expected_total = sum(float(task["estimated_seconds"]) for task in tasks)
    expected_margin = POD_LIMIT_SECONDS - expected_total
    if expected_margin <= 60 * 60:
        raise RuntimeError(
            "Combined plan must retain at least one hour below the 600-minute limit"
        )

    status_path = output_root / "combined_batch_status.json"
    summary_path = output_root / "combined_batch_summary.json"
    records: list[dict[str, Any]] = []
    batch_start = time.time()

    log("=" * 80)
    log("COMBINED FULL BATCH / CHAOYANG + FLOWERS-102 + CIFAR-100")
    log("=" * 80)
    log("[ORDER] Chaoyang five -> Flowers five -> CIFAR-100 one")
    log(f"[CIFAR_METHOD] {args.cifar_method}")
    log(f"[PATH] output_root={output_root}")
    log(
        f"[RUNTIME_PLAN] expected_total={format_duration(expected_total)} "
        f"pod_limit={format_duration(POD_LIMIT_SECONDS)} "
        f"expected_margin={format_duration(expected_margin)}"
    )
    log(
        "[OUTPUT] Every dataset and method has an independent /app/output "
        "directory. Completed results remain if a later task fails."
    )

    try:
        for order, task in enumerate(tasks, start=1):
            task_output = Path(task["output_dir"])
            command = [
                sys.executable,
                str(REPOSITORY_ROOT / "methods/run_five_methods.py"),
                *task["arguments"],
                "--output-dir",
                str(task_output),
            ]
            record: dict[str, Any] = {
                "order": order,
                "dataset": task["dataset"],
                "methods": task["methods"],
                "status": "running",
                "estimated_seconds": task["estimated_seconds"],
                "estimated_human": format_duration(task["estimated_seconds"]),
                "output_dir": str(task_output),
                "command": command,
            }
            records.append(record)
            atomic_json(
                status_path,
                {
                    "status": "running",
                    "expected_total_seconds": expected_total,
                    "records": records,
                },
            )
            log("=" * 80)
            log(
                f"[BATCH][{order}/{len(tasks)}] START "
                f"dataset={task['dataset']} methods={','.join(task['methods'])} "
                f"estimate={record['estimated_human']}"
            )
            task_start = time.time()
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
            task_summary = task_output / "five_method_summary.json"
            if not task_summary.is_file():
                raise FileNotFoundError(
                    f"Missing task summary for {task['dataset']}: {task_summary}"
                )
            payload = json.loads(task_summary.read_text(encoding="utf-8"))
            record["status"] = "complete"
            record["elapsed_seconds"] = time.time() - task_start
            record["elapsed_human"] = format_duration(record["elapsed_seconds"])
            record["summary"] = str(task_summary)
            record["results"] = payload["records"]
            atomic_json(
                status_path,
                {
                    "status": "running" if order < len(tasks) else "complete",
                    "expected_total_seconds": expected_total,
                    "records": records,
                },
            )
            log(
                f"[BATCH][{order}/{len(tasks)}] DONE "
                f"dataset={task['dataset']} elapsed={record['elapsed_human']}"
            )
    except Exception as error:
        if records and records[-1]["status"] == "running":
            records[-1]["status"] = "failed"
            records[-1]["error"] = f"{type(error).__name__}: {error}"
        atomic_json(
            status_path,
            {
                "status": "failed",
                "expected_total_seconds": expected_total,
                "records": records,
            },
        )
        raise

    elapsed = time.time() - batch_start
    final_summary = {
        "status": "complete",
        "cifar_method": args.cifar_method,
        "expected_total_seconds": expected_total,
        "expected_total_human": format_duration(expected_total),
        "pod_limit_seconds": POD_LIMIT_SECONDS,
        "expected_margin_seconds": expected_margin,
        "expected_margin_human": format_duration(expected_margin),
        "elapsed_seconds": elapsed,
        "elapsed_human": format_duration(elapsed),
        "records": records,
    }
    atomic_json(summary_path, final_summary)

    log("=" * 80)
    for record in records:
        log(
            f"[FINAL_RESULT] dataset={record['dataset']} "
            f"methods={','.join(record['methods'])} "
            f"elapsed={record['elapsed_human']} output={record['output_dir']}"
        )
    log(f"[FINAL_RESULT] combined_summary={summary_path}")
    log(
        f"[FINAL_RESULT] elapsed={format_duration(elapsed)} "
        f"expected={format_duration(expected_total)}"
    )
    log(
        "[DONE] Combined Flowers-102, Chaoyang, and CIFAR-100 full batch "
        "completed successfully; resources may be released."
    )


def cli_main() -> None:
    try:
        main()
    except Exception as error:
        log("=" * 80)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Combined full batch did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
