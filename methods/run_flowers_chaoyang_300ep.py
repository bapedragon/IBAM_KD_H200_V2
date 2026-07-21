#!/usr/bin/env python3
"""Run all generic KD baselines for 300 epochs on Chaoyang and Flowers."""

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
DATASETS = ("chaoyang", "flowers102")
METHODS = ("KD", "CRD", "ReviewKD", "MGD", "OFA")
PLANNED_EPOCHS = 300
POD_LIMIT_SECONDS = 600 * 60
SAFETY_MARGIN_SECONDS = 60 * 60

# Exact averages from the already completed H200 full runs.  Only the epoch
# count is scaled; method losses and every other dataset protocol value remain
# unchanged.
MEASURED_AVG_EPOCH_SECONDS = {
    "chaoyang": {
        "KD": 8.231867101192474,
        "CRD": 9.038858015537262,
        "ReviewKD": 8.792574033737182,
        "MGD": 8.470481848716735,
        "OFA": 9.767858831882476,
    },
    "flowers102": {
        "KD": 10.58735299229622,
        "CRD": 10.988087257146836,
        "ReviewKD": 10.740112318992615,
        "MGD": 11.347939156293869,
        "OFA": 11.696486096382142,
    },
}


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


def estimated_dataset_seconds(dataset: str) -> float:
    return sum(MEASURED_AVG_EPOCH_SECONDS[dataset].values()) * PLANNED_EPOCHS


def estimated_total_seconds() -> float:
    return sum(estimated_dataset_seconds(dataset) for dataset in DATASETS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=REPOSITORY_ROOT / "teachers/checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/output/generic_kd_flowers_chaoyang_300ep_seed42"),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_tasks(args: argparse.Namespace, output_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "order": order,
            "dataset": dataset,
            "methods": list(METHODS),
            "estimated_seconds": estimated_dataset_seconds(dataset),
            "output_dir": output_root / dataset,
            "command": [
                sys.executable,
                str(REPOSITORY_ROOT / "methods/run_five_methods.py"),
                "--dataset",
                dataset,
                "--full-run",
                "--planned-epochs",
                str(PLANNED_EPOCHS),
                "--teacher-root",
                str(args.teacher_root),
                "--output-dir",
                str(output_root / dataset),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
            ],
        }
        for order, dataset in enumerate(DATASETS, start=1)
    ]


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    estimate = estimated_total_seconds()
    margin = POD_LIMIT_SECONDS - estimate
    if margin < SAFETY_MARGIN_SECONDS:
        raise RuntimeError(
            "The measured 300-epoch plan does not retain the required one-hour "
            f"margin: estimate={format_duration(estimate)} "
            f"margin={format_duration(margin)}"
        )

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "generic_kd_300ep_status.json"
    summary_path = output_root / "generic_kd_300ep_summary.json"
    tasks = build_tasks(args, output_root)
    records: list[dict[str, Any]] = []
    sequence_start = time.time()

    log("=" * 80)
    log("GENERIC KD 300-EPOCH BATCH / CHAOYANG -> FLOWERS-102")
    log("=" * 80)
    log(f"[METHODS] {','.join(METHODS)}")
    log(f"[EPOCH_PLAN] both_datasets={PLANNED_EPOCHS} epoch_only_change=True")
    log(
        "[PROTOCOL_LOCK] Per-dataset batch, warm-up, augmentation, seed, "
        "teacher, adapters, and method loss coefficients are unchanged."
    )
    log(
        f"[RUNTIME_PLAN] estimated_total={format_duration(estimate)} "
        f"pod_limit={format_duration(POD_LIMIT_SECONDS)} "
        f"margin={format_duration(margin)}"
    )
    for dataset in DATASETS:
        log(
            f"[RUNTIME_PLAN] dataset={dataset} "
            f"estimated_five_methods={format_duration(estimated_dataset_seconds(dataset))}"
        )
    log(
        "[OUTPUT] Each dataset and method has an independent directory; "
        "completed artifacts remain if a later subprocess fails."
    )

    try:
        for task in tasks:
            record = {
                "order": task["order"],
                "dataset": task["dataset"],
                "methods": task["methods"],
                "planned_epochs": PLANNED_EPOCHS,
                "estimated_seconds": task["estimated_seconds"],
                "estimated_human": format_duration(task["estimated_seconds"]),
                "output_dir": str(task["output_dir"]),
                "command": task["command"],
                "status": "running",
            }
            records.append(record)
            atomic_json(status_path, {"status": "running", "records": records})
            log("=" * 80)
            log(
                f"[DATASET_SEQUENCE][{task['order']}/{len(tasks)}] "
                f"START dataset={task['dataset']}"
            )
            task_start = time.time()
            subprocess.run(task["command"], cwd=REPOSITORY_ROOT, check=True)
            record["status"] = "complete"
            record["elapsed_seconds"] = time.time() - task_start
            record["summary"] = str(
                Path(task["output_dir"]) / "five_method_summary.json"
            )
            if not Path(record["summary"]).is_file():
                raise FileNotFoundError(
                    f"Missing dataset summary after completion: {record['summary']}"
                )
            atomic_json(
                status_path,
                {
                    "status": (
                        "complete" if task["order"] == len(tasks) else "running"
                    ),
                    "records": records,
                },
            )
            log(
                f"[DATASET_SEQUENCE][{task['order']}/{len(tasks)}] "
                f"DONE dataset={task['dataset']}"
            )
    except Exception as error:
        if records and records[-1]["status"] == "running":
            records[-1]["status"] = "failed"
            records[-1]["error"] = f"{type(error).__name__}: {error}"
        atomic_json(status_path, {"status": "failed", "records": records})
        raise

    elapsed = time.time() - sequence_start
    atomic_json(
        summary_path,
        {
            "status": "complete",
            "datasets": list(DATASETS),
            "methods": list(METHODS),
            "planned_epochs": PLANNED_EPOCHS,
            "epoch_only_change": True,
            "estimated_total_seconds": estimate,
            "estimated_total_human": format_duration(estimate),
            "pod_limit_seconds": POD_LIMIT_SECONDS,
            "safety_margin_seconds": margin,
            "safety_margin_human": format_duration(margin),
            "elapsed_seconds": elapsed,
            "elapsed_human": format_duration(elapsed),
            "records": records,
        },
    )
    log("=" * 80)
    log(f"[FINAL_RESULT] completed_datasets={','.join(DATASETS)}")
    log(f"[FINAL_RESULT] summary={summary_path}")
    log(
        "[DONE] All ten 300-epoch generic KD runs completed successfully; "
        "resources may be released."
    )


def cli_main() -> None:
    try:
        main()
    except Exception as error:
        log("=" * 80)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Generic KD 300-epoch batch did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
