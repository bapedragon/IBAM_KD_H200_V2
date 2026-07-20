#!/usr/bin/env python3
"""Time all five KD methods on Flowers-102 and Chaoyang in one H200 job."""

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
DATASETS = ("flowers102", "chaoyang")


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
    parser.add_argument("--teacher-root", type=Path, default=REPOSITORY_ROOT / "teachers/checkpoints")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/flowers_chaoyang_timing_v2"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "two_dataset_timing_status.json"
    summary_path = output_root / "two_dataset_timing_summary.json"
    dataset_records: list[dict[str, Any]] = []
    start = time.time()

    log("=" * 80)
    log("FLOWERS-102 + CHAOYANG / FIVE-METHOD TIMING")
    log("=" * 80)
    log("[METHODS] KD,CRD,ReviewKD,MGD,OFA")
    log("[DATASETS] flowers102,chaoyang")
    log("[MODE] two full-data epochs per dataset and method")
    log(f"[PATH] output_root={output_root}")
    log(
        "[OUTPUT] Ten independent method directories plus per-dataset and "
        "combined timing summaries will be preserved."
    )

    try:
        for order, dataset in enumerate(DATASETS, start=1):
            dataset_output = output_root / dataset
            command = [
                sys.executable,
                str(REPOSITORY_ROOT / "methods/run_five_methods.py"),
                "--dataset",
                dataset,
                "--timing-run",
                "--teacher-root",
                str(args.teacher_root),
                "--output-dir",
                str(dataset_output),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
            ]
            record: dict[str, Any] = {
                "order": order,
                "dataset": dataset,
                "status": "running",
                "output_dir": str(dataset_output),
                "command": command,
            }
            dataset_records.append(record)
            atomic_json(
                status_path,
                {"status": "running", "datasets": dataset_records},
            )
            log("=" * 80)
            log(f"[DATASET_SEQUENCE][{order}/2] START dataset={dataset}")
            dataset_start = time.time()
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
            per_dataset_summary = dataset_output / "five_method_summary.json"
            if not per_dataset_summary.is_file():
                raise FileNotFoundError(
                    f"Missing {dataset} timing summary: {per_dataset_summary}"
                )
            payload = json.loads(per_dataset_summary.read_text(encoding="utf-8"))
            record["status"] = "complete"
            record["elapsed_seconds"] = time.time() - dataset_start
            record["summary"] = str(per_dataset_summary)
            record["estimated_full_seconds"] = payload[
                "estimated_selected_methods_full_seconds"
            ]
            record["estimated_full_human"] = payload[
                "estimated_selected_methods_full_human"
            ]
            record["methods"] = payload["records"]
            atomic_json(
                status_path,
                {
                    "status": "running" if order < len(DATASETS) else "complete",
                    "datasets": dataset_records,
                },
            )
            log(
                f"[DATASET_SEQUENCE][{order}/2] DONE dataset={dataset} "
                f"estimated_five_method_full={record['estimated_full_human']}"
            )
    except Exception as error:
        if dataset_records and dataset_records[-1]["status"] == "running":
            dataset_records[-1]["status"] = "failed"
            dataset_records[-1]["error"] = f"{type(error).__name__}: {error}"
        atomic_json(
            status_path,
            {"status": "failed", "datasets": dataset_records},
        )
        raise

    elapsed = time.time() - start
    grand_total = sum(
        float(record["estimated_full_seconds"]) for record in dataset_records
    )
    final_summary = {
        "status": "complete",
        "mode": "timing",
        "datasets": list(DATASETS),
        "methods": ["KD", "CRD", "ReviewKD", "MGD", "OFA"],
        "elapsed_seconds": elapsed,
        "elapsed_human": format_duration(elapsed),
        "estimated_all_full_seconds": grand_total,
        "estimated_all_full_human": format_duration(grand_total),
        "dataset_records": dataset_records,
    }
    atomic_json(summary_path, final_summary)

    log("=" * 80)
    for dataset_record in dataset_records:
        for method_record in dataset_record["methods"]:
            log(
                f"[TIMING_MATRIX] dataset={dataset_record['dataset']} "
                f"method={method_record['method']} "
                f"avg_epoch={float(method_record['avg_epoch_seconds']):.1f}s "
                f"planned_epochs={method_record['planned_epochs']} "
                f"estimated_full={method_record['estimated_planned_human']}"
            )
        log(
            f"[TIMING_DATASET] dataset={dataset_record['dataset']} "
            f"estimated_five_method_full={dataset_record['estimated_full_human']}"
        )
    log(
        f"[TIMING_TOTAL] estimated_ten_runs_full={format_duration(grand_total)} "
        f"({grand_total:.1f}s)"
    )
    log(f"[FINAL_RESULT] combined_summary={summary_path}")
    log(
        "[DONE] Flowers-102 and Chaoyang five-method timing completed "
        "successfully; resources may be released."
    )


def cli_main() -> None:
    try:
        main()
    except Exception as error:
        log("=" * 80)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Two-dataset timing did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
