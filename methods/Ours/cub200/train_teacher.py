#!/usr/bin/env python3
"""Train the scratch 32x32 ResNet56 guidance teacher for CUB-200-2011."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.Ours.cub200.dataset import (  # noqa: E402
    CUB200Dataset,
    EXPECTED_TEST_IMAGES,
    EXPECTED_TRAIN_IMAGES,
    ensure_cub200,
)
from teachers import train_teacher_cifar100 as common  # noqa: E402


NUM_CLASSES = 200
IMAGE_SIZE = 32
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 200
PLANNED_EPOCHS = 300
BASE_LR = 0.1
WEIGHT_DECAY = 5e-4
SEED = 1
RECIPE_NAME = "cub200_official_split_resnet56_32_scratch_300ep_seed1"


class ResNet56CUB200(common.ResNet56):
    """CIFAR-style ResNet56 with a 200-class CUB classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.head.fc = nn.Linear(64, NUM_CLASSES, bias=True)
        self._init_weights(self.head.fc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--run-name", default=None)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--timing-run", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke-train-samples", type=int, default=512)
    parser.add_argument("--smoke-test-samples", type=int, default=512)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require an already extracted CUB_200_2011 directory.",
    )
    return parser.parse_args()


def deterministic_subset(dataset: Dataset[Any], size: int, seed: int) -> Dataset[Any]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[
        : min(size, len(dataset))
    ]
    return Subset(dataset, indices.tolist())


def build_loaders(
    args: argparse.Namespace, device: torch.device
) -> tuple[DataLoader[Any], DataLoader[Any], Path]:
    dataset_root = ensure_cub200(args.data_dir, download=not args.no_download)
    train_dataset: Dataset[Any] = CUB200Dataset(
        dataset_root,
        split="train",
        transform=common.OfficialLGTrainTransform(),
    )
    test_dataset: Dataset[Any] = CUB200Dataset(
        dataset_root,
        split="test",
        transform=common.official_test_transform(),
    )
    if len(train_dataset) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError(f"Unexpected CUB train size: {len(train_dataset)}")
    if len(test_dataset) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(f"Unexpected CUB test size: {len(test_dataset)}")
    if args.smoke:
        train_dataset = deterministic_subset(
            train_dataset, args.smoke_train_samples, SEED
        )
        test_dataset = deterministic_subset(
            test_dataset, args.smoke_test_samples, SEED + 1
        )
    shared = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **shared,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **shared,
    )
    common.log(
        f"[DATA] root={dataset_root} train_samples={len(train_dataset)} "
        f"test_samples={len(test_dataset)} split=official_train/test"
    )
    return train_loader, test_loader, dataset_root


def protocol_check(model: nn.Module) -> None:
    model.eval()
    with torch.inference_mode():
        features = model.forward_features(torch.zeros(2, 3, 32, 32))
        logits = model.head(features[-1])
    feature_shapes = tuple(tuple(feature.shape) for feature in features)
    expected_features = (
        (2, 16, 32, 32),
        (2, 32, 16, 16),
        (2, 64, 8, 8),
    )
    if feature_shapes != expected_features or tuple(logits.shape) != (2, 200):
        raise RuntimeError(
            "Teacher protocol check failed: "
            f"features={feature_shapes} logits={tuple(logits.shape)}"
        )
    common.log(
        "[PROTOCOL_CHECK] status=PASS "
        f"features={feature_shapes} logits={tuple(logits.shape)}"
    )


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    accuracy: float,
    best_accuracy: float,
    epoch_times: list[float],
    mode: str,
) -> dict[str, Any]:
    state = model.state_dict()
    return {
        "epoch": epoch,
        "accuracy": accuracy,
        "best_accuracy": best_accuracy,
        "model": state,
        "model_state": state,
        "optimizer_state": optimizer.state_dict(),
        "model_name": "ResNet56",
        "architecture": "CIFAR-style ResNet56 (6n+2, n=9)",
        "dataset": "cub200",
        "num_classes": NUM_CLASSES,
        "input_resolution": IMAGE_SIZE,
        "train_split": "official_train",
        "evaluation_split": "official_test",
        "recipe_name": RECIPE_NAME,
        "pretrained": False,
        "epoch_times": epoch_times,
        "mode": mode,
    }


def write_manifest(run_dir: Path, best_path: Path, payload: dict[str, Any]) -> Path:
    manifest_path = run_dir / "manifest.json"
    spec = {
        "selected_kind": "best",
        "checkpoint": best_path.name,
        "sha256": common.sha256_file(best_path),
        "epoch": int(payload["epoch"]),
        "top1": float(payload["accuracy"]),
        "num_classes": NUM_CLASSES,
        "input_resolution": IMAGE_SIZE,
        "recipe_name": RECIPE_NAME,
        "pretrained": False,
    }
    common.atomic_json_save(
        {"version": 1, "teachers": {"cub200": spec}}, manifest_path
    )
    return manifest_path


def train(args: argparse.Namespace) -> None:
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.smoke_train_samples <= 0 or args.smoke_test_samples <= 0:
        raise ValueError("Smoke sample counts must be positive")
    common.install_signal_handlers()
    common.seed_everything(SEED)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "smoke" if args.smoke else "timing" if args.timing_run else "full"
    epochs = 1 if args.smoke else 2 if args.timing_run else PLANNED_EPOCHS
    run_name = args.run_name or f"teacher_{RECIPE_NAME}_{mode}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "teacher_resnet56_cub200_32_best.pt"
    latest_path = run_dir / "teacher_resnet56_cub200_32_latest.pt"
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"

    common.log("=" * 80)
    common.log("TRAIN CUB-200-2011 RESNET56 TEACHER (SCRATCH, 32 x 32)")
    common.log("=" * 80)
    common.log(
        f"[MODE] mode={mode} actual_epochs={epochs} "
        f"planned_epochs={PLANNED_EPOCHS}"
    )
    common.log(f"[PATH] run_dir={run_dir.resolve()}")
    common.log(
        "[PROTOCOL] official_split train=5994 test=5794 pretrained=False "
        "optimizer=SGD lr=0.1 momentum=0.9 nesterov=True "
        "weight_decay=0.0005 cosine=300ep seed=1 fp32=True"
    )
    common.log(
        "[LEAKAGE_GUARD] ImageNet-pretrained weights are disabled because the "
        "official CUB page warns that CUB images overlap with ImageNet."
    )

    model = ResNet56CUB200()
    protocol_check(model)
    model.to(device)
    train_loader, test_loader, dataset_root = build_loaders(args, device)
    optimizer = torch.optim.SGD(
        common.parameter_groups(model),
        lr=BASE_LR,
        momentum=0.9,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PLANNED_EPOCHS, eta_min=0.0
    )
    criterion = nn.CrossEntropyLoss()
    scaler = common.create_grad_scaler(False)

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        csv.writer(metrics_file).writerow(
            (
                "epoch",
                "train_loss",
                "train_top1",
                "test_top1",
                "best_top1",
                "learning_rate",
                "epoch_seconds",
            )
        )

    best_accuracy = -1.0
    latest_accuracy = -1.0
    epoch_times: list[float] = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        learning_rate = optimizer.param_groups[0]["lr"]
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = targets.size(0)
            total_loss += float(loss.detach()) * batch_size
            correct += common.top1_correct(logits.detach(), targets)
            total += batch_size

        latest_accuracy = common.evaluate(model, test_loader, device, False)
        scheduler.step()
        epoch_seconds = time.time() - epoch_start
        epoch_times.append(epoch_seconds)
        is_best = latest_accuracy > best_accuracy
        if is_best:
            best_accuracy = latest_accuracy
        payload = checkpoint_payload(
            model,
            optimizer,
            epoch=epoch,
            accuracy=latest_accuracy,
            best_accuracy=best_accuracy,
            epoch_times=epoch_times,
            mode=mode,
        )
        common.atomic_torch_save(payload, latest_path)
        if is_best:
            common.atomic_torch_save(payload, best_path)
        with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
            csv.writer(metrics_file).writerow(
                (
                    epoch,
                    f"{total_loss / max(1, total):.8f}",
                    f"{100.0 * correct / max(1, total):.6f}",
                    f"{latest_accuracy:.6f}",
                    f"{best_accuracy:.6f}",
                    f"{learning_rate:.10f}",
                    f"{epoch_seconds:.6f}",
                )
            )
        average_epoch = sum(epoch_times) / len(epoch_times)
        common.log(
            f"[TEACHER][{epoch:03d}/{epochs:03d}] "
            f"loss={total_loss / max(1, total):.4f} "
            f"train_acc={100.0 * correct / max(1, total):.2f}% "
            f"test_acc={latest_accuracy:.2f}% best={best_accuracy:.2f}% "
            f"lr={learning_rate:.8f} time={epoch_seconds:.1f}s "
            f"est_300={common.format_duration(average_epoch * PLANNED_EPOCHS)}"
            + (" saved_best" if is_best else "")
        )

    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    manifest_path = write_manifest(run_dir, best_path, best_payload)
    elapsed = time.time() - start
    average_epoch = sum(epoch_times) / len(epoch_times)
    summary = {
        "status": "complete",
        "mode": mode,
        "dataset": "cub200",
        "dataset_root": str(dataset_root),
        "pretrained": False,
        "completed_epoch": epochs,
        "planned_epochs": PLANNED_EPOCHS,
        "latest_top1": latest_accuracy,
        "best_top1": best_accuracy,
        "avg_epoch_seconds": average_epoch,
        "estimated_planned_seconds": average_epoch * PLANNED_EPOCHS,
        "estimated_planned_human": common.format_duration(
            average_epoch * PLANNED_EPOCHS
        ),
        "elapsed_seconds": elapsed,
        "paths": {
            "best": str(best_path.resolve()),
            "latest": str(latest_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "metrics": str(metrics_path.resolve()),
        },
        "sha256": {"best": common.sha256_file(best_path)},
    }
    common.atomic_json_save(summary, summary_path)
    common.log(
        f"[FINAL_RESULT] teacher_best_top1={best_accuracy:.2f}% "
        f"checkpoint={best_path.resolve()}"
    )
    common.log(f"[FINAL_RESULT] manifest={manifest_path.resolve()}")
    common.log("[DONE] CUB-200 teacher training completed successfully.")


def main() -> None:
    try:
        train(parse_args())
    except Exception as error:
        common.log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
