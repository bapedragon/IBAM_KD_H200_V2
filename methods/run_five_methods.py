#!/usr/bin/env python3
"""Run the five generic KD methods sequentially for one fixed dataset."""

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
    ("KD", "KD", "kd"),
    ("CRD", "CRD", "crd"),
    ("ReviewKD", "ReviewKD", "reviewkd"),
    ("MGD", "MGD", "mgd"),
    ("OFA", "OFA", "ofa"),
)
DATASET_CONFIGS = {
    "cifar100": {
        "planned_epochs": 300,
        "batch_size": 128,
        "warmup_epochs": 20,
        "default_data_dir": Path("./data"),
    },
    "flowers102": {
        "planned_epochs": 200,
        "batch_size": 64,
        "warmup_epochs": 5,
        "default_data_dir": Path("./data"),
    },
    "chaoyang": {
        "planned_epochs": 100,
        "batch_size": 64,
        "warmup_epochs": 5,
        "default_data_dir": Path("/app/data/chaoyang"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_CONFIGS), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--timing-run",
        action="store_true",
        help="Run two full-data epochs per selected method with the full schedule.",
    )
    mode.add_argument(
        "--full-run",
        action="store_true",
        help="Run the selected methods for the dataset's planned epochs.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[method for method, _, _ in METHODS],
        default=[method for method, _, _ in METHODS],
        help=(
            "Subset executed in canonical KD -> CRD -> ReviewKD -> MGD -> OFA "
            "order. Defaults to all five."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=REPOSITORY_ROOT / "teachers/checkpoints",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_name(prefix: str, dataset: str, timing_run: bool, epochs: int, seed: int) -> str:
    suffix = "timing_2ep" if timing_run else f"full_{epochs}ep"
    return f"{prefix}_{dataset}_deit_ti_{suffix}_seed{seed}_v2"


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    requested = set(args.methods)
    selected_methods = tuple(item for item in METHODS if item[0] in requested)
    selected_names = [item[0] for item in selected_methods]
    if len(selected_names) != len(args.methods):
        raise ValueError("--methods must not contain duplicates")

    config = DATASET_CONFIGS[args.dataset]
    data_dir = args.data_dir or config["default_data_dir"]
    planned_epochs = int(config["planned_epochs"])
    batch_size = int(config["batch_size"])
    warmup_epochs = int(config["warmup_epochs"])
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "five_method_status.json"
    summary_path = output_root / "five_method_summary.json"
    records: list[dict[str, Any]] = []
    sequence_start = time.time()

    log("=" * 80)
    log(
        f"{args.dataset.upper()} DEIT-TI SEQUENTIAL DISTILLATION: "
        + " -> ".join(selected_names)
    )
    log("=" * 80)
    log(f"[MODE] timing_run={args.timing_run} full_run={args.full_run}")
    log(f"[METHODS] selected={','.join(selected_names)}")
    log(f"[PATH] repository={REPOSITORY_ROOT}")
    log(f"[PATH] data_dir={data_dir.expanduser().resolve()}")
    log(f"[PATH] teacher_root={args.teacher_root.resolve()}")
    log(f"[PATH] output_root={output_root}")
    log(
        f"[PROTOCOL] dataset={args.dataset} student=DeiT-Ti student_input=224 "
        f"teacher=ResNet56 teacher_input=32 epochs={planned_epochs} "
        f"batch={batch_size} warmup={warmup_epochs} seed={args.seed}"
    )
    log(
        "[OUTPUT] Every method uses an independent run directory; completed "
        "artifacts remain intact if a later method fails."
    )

    verify_command = [
        sys.executable,
        str(REPOSITORY_ROOT / "teachers/verify_checkpoints.py"),
        "--dataset",
        args.dataset,
        "--checkpoint-root",
        str(args.teacher_root),
    ]
    log(f"[PRECHECK] command={' '.join(verify_command)}")
    subprocess.run(verify_command, cwd=REPOSITORY_ROOT, check=True)

    total_methods = len(selected_methods)
    try:
        for index, (method, directory, prefix) in enumerate(
            selected_methods, start=1
        ):
            name = run_name(
                prefix,
                args.dataset,
                args.timing_run,
                planned_epochs,
                args.seed,
            )
            run_dir = output_root / name
            command = [
                sys.executable,
                str(REPOSITORY_ROOT / f"methods/{directory}/{args.dataset}/train.py"),
                "--protocol-name",
                f"{args.dataset}_deit_ti_common_kd_v2",
                "--data-dir",
                str(data_dir),
                "--teacher-root",
                str(args.teacher_root),
                "--output-dir",
                str(output_root),
                "--run-name",
                name,
                "--student-epochs",
                str(planned_epochs),
                "--batch-size",
                str(batch_size),
                "--num-workers",
                str(args.num_workers),
                "--image-size",
                "224",
                "--warmup-epochs",
                str(warmup_epochs),
                "--seed",
                str(args.seed),
            ]
            if args.timing_run:
                command.append("--timing-run")

            record: dict[str, Any] = {
                "order": index,
                "method": method,
                "status": "running",
                "run_dir": str(run_dir),
                "command": command,
            }
            records.append(record)
            atomic_json(
                status_path,
                {
                    "status": "running",
                    "dataset": args.dataset,
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
            method_summary_path = run_dir / "summary.json"
            if not method_summary_path.is_file():
                raise FileNotFoundError(
                    f"{method} completed without summary: {method_summary_path}"
                )
            summary = json.loads(method_summary_path.read_text(encoding="utf-8"))
            record["summary"] = str(method_summary_path)
            record["best_top1"] = summary.get("best_top1")
            record["avg_epoch_seconds"] = summary.get("avg_epoch_seconds")
            record["planned_epochs"] = summary.get("planned_epochs")
            record["estimated_planned_seconds"] = summary.get(
                "estimated_planned_seconds"
            )
            record["estimated_planned_human"] = summary.get(
                "estimated_planned_human"
            )
            atomic_json(
                status_path,
                {
                    "status": "running" if index < total_methods else "complete",
                    "dataset": args.dataset,
                    "mode": "timing" if args.timing_run else "full",
                    "selected_methods": selected_names,
                    "records": records,
                },
            )
            log(
                f"[SEQUENCE][{index}/{total_methods}] DONE method={method} "
                f"best_top1={record['best_top1']} "
                f"estimated_full={record['estimated_planned_human']}"
            )
    except Exception as error:
        if records and records[-1]["status"] == "running":
            records[-1]["status"] = "failed"
            records[-1]["error"] = f"{type(error).__name__}: {error}"
        atomic_json(
            status_path,
            {
                "status": "failed",
                "dataset": args.dataset,
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
        "dataset": args.dataset,
        "student": "deit_ti",
        "teacher_input": 32,
        "student_input": 224,
        "planned_epochs": planned_epochs,
        "batch_size": batch_size,
        "warmup_epochs": warmup_epochs,
        "selected_methods": selected_names,
        "elapsed_seconds": elapsed,
        "elapsed_human": format_duration(elapsed),
        "estimated_selected_methods_full_seconds": estimated_full_seconds,
        "estimated_selected_methods_full_human": format_duration(
            estimated_full_seconds
        ),
        "records": records,
    }
    atomic_json(summary_path, final_summary)

    log("=" * 80)
    log(f"[FINAL_RESULT] completed_methods={','.join(selected_names)}")
    for record in records:
        log(
            f"[FINAL_RESULT] method={record['method']} "
            f"avg_epoch={float(record['avg_epoch_seconds']):.1f}s "
            f"planned_epochs={record['planned_epochs']} "
            f"estimated_full={record['estimated_planned_human']}"
        )
    log(f"[FINAL_RESULT] aggregate_summary={summary_path}")
    log(
        f"[TIMING] estimated_selected_methods_full="
        f"{format_duration(estimated_full_seconds)} ({estimated_full_seconds:.1f}s)"
    )
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
        log("[FATAL] Five-method sequence did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
