#!/usr/bin/env python3
"""Run method-separated ALG then Ours on Flowers train+val/test."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    ("ALG", Path("methods/ALG/flowers102/train_official_split.py")),
    ("Ours", Path("methods/Ours/flowers102/train_official_split.py")),
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
    mode.add_argument("--timing-run", action="store_true")
    mode.add_argument("--full-run", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=REPOSITORY_ROOT / "teachers/checkpoints",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    default_output = (
        Path("/tmp/flowers102_trainval_test_alg_ours_timing")
        if args.timing_run
        else Path("/app/output/flowers102_trainval_test_alg_ours_300ep_seed1")
    )
    output_root = (args.output_dir or default_output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "sequence_status.json"
    summary_path = output_root / "sequence_summary.json"
    records: list[dict[str, Any]] = []
    sequence_start = time.time()

    log("=" * 80)
    log("FLOWERS-102 TRAIN+VAL/TEST, METHOD-SEPARATED: ALG -> OURS")
    log("=" * 80)
    log(f"[MODE] timing_run={args.timing_run} full_run={args.full_run}")
    log("[SPLIT_LOCK] train=official_train+val=2040 eval=official_test=6149")
    log("[SELECTION_LOCK] best=official_test_top1")
    log("[ALG_PROTOCOL] ALG paper + public LG code; no Ours settings")
    log("[ALG_PROTOCOL] epochs=300 train/eval_batch=128/200 controller_warmup=0")
    log("[ALG_SPLIT] matches paper dataset accounting: train=2040 test=6149")
    log("[OURS_PROTOCOL] Ours paper + supplied Ours source; ALG only fills gaps")
    log("[OURS_PROTOCOL] epochs=300 train/eval_batch=128/200 controller_warmup=20")
    log(f"[PATH] output_root={output_root}")

    try:
        for order, (method, relative_script) in enumerate(TASKS, start=1):
            mode_name = "timing_2ep" if args.timing_run else "300ep"
            run_name = (
                f"{method.lower()}_flowers102_deit_ti_trainval_test_"
                f"{mode_name}_seed1"
            )
            task_output = output_root / method / "flowers102"
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
                "dataset": "flowers102",
                "status": "running",
                "run_dir": str(run_dir),
                "command": command,
            }
            records.append(record)
            atomic_json(status_path, {"status": "running", "records": records})
            log("=" * 80)
            log(f"[SEQUENCE][{order}/2] START method={method}")
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
            task_summary_path = run_dir / "summary.json"
            if not task_summary_path.is_file():
                raise FileNotFoundError(f"Missing task summary: {task_summary_path}")
            task_summary = json.loads(task_summary_path.read_text(encoding="utf-8"))
            best_test_top1 = task_summary.get("selection_best_top1")
            if best_test_top1 is None:
                raise KeyError(
                    f"Missing selection_best_top1 in {task_summary_path}"
                )
            record.update(
                {
                    "status": "complete",
                    "summary": str(task_summary_path),
                    "best_test_top1": float(best_test_top1),
                    "estimated_planned_seconds": task_summary.get(
                        "estimated_planned_seconds"
                    ),
                }
            )
            atomic_json(
                status_path,
                {"status": "complete" if order == 2 else "running", "records": records},
            )
            log(
                f"[SEQUENCE][{order}/2] DONE method={method} "
                f"best_test_top1={float(record['best_test_top1']):.2f}%"
            )
    except Exception as error:
        if records and records[-1]["status"] == "running":
            records[-1]["status"] = "failed"
            records[-1]["error"] = f"{type(error).__name__}: {error}"
        atomic_json(status_path, {"status": "failed", "records": records})
        raise

    estimated_full_seconds = sum(
        float(record.get("estimated_planned_seconds") or 0.0)
        for record in records
    )
    elapsed = time.time() - sequence_start
    payload = {
        "status": "complete",
        "protocol": "method_separated_v2_trainval_test_300ep_seed1",
        "split": {"train_plus_val": 2040, "test": 6149},
        "checkpoint_selection": "official_test_top1",
        "final_report": "best_official_test_top1",
        "elapsed_seconds": elapsed,
        "elapsed_human": format_duration(elapsed),
        "estimated_full_seconds": estimated_full_seconds,
        "estimated_full_human": format_duration(estimated_full_seconds),
        "pod_limit_passed": estimated_full_seconds < POD_LIMIT_SECONDS,
        "records": records,
    }
    atomic_json(summary_path, payload)
    by_method = {record["method"]: record for record in records}
    log("=" * 80)
    log(
        f"[FINAL_RESULT] completed_tasks=2/2 "
        f"estimated_full={format_duration(estimated_full_seconds)}"
    )
    for method in ("ALG", "Ours"):
        record = by_method[method]
        log(
            f"[FINAL_BEST][{method}] "
            f"best_test_top1={float(record['best_test_top1']):.2f}%"
        )
    alg_test = by_method["ALG"].get("best_test_top1")
    ours_test = by_method["Ours"].get("best_test_top1")
    if alg_test is not None and ours_test is not None:
        log(
            f"[FINAL_COMPARISON] ours_minus_alg_test="
            f"{float(ours_test) - float(alg_test):+.2f}pp"
        )
    log(
        f"[POD_LIMIT_CHECK] status="
        f"{'PASS' if payload['pod_limit_passed'] else 'FAIL'} limit=10h 00m 00s"
    )
    log(f"[FINAL_RESULT] summary={summary_path}")
    log("[DONE] Flowers train+val/test ALG and Ours completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"[FATAL] {type(error).__name__}: {error}")
        raise
