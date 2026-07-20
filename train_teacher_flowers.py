#!/usr/bin/env python3
"""Train the adjusted Flowers-102 ResNet56 guidance teacher at 32 x 32.

Recipe v2 keeps the first run's 300-epoch schedule and all paper-confirmed
teacher constraints while changing only augmentation to the public LG code's
weak-augmentation path. On the KAU H200 runner, pass ``--output-dir
/app/output`` for artifacts that must survive Pod release. Timing-run artifacts
remain in the temporary cloned repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import Flowers102
from torchvision.datasets.utils import check_integrity, extract_archive

import train_teacher_cifar100 as common


NUM_CLASSES = 102
IMAGE_SIZE = 32
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 200
PLANNED_EPOCHS = 300
BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
SEED = 1
REFERENCE_TEACHER_TOP1 = 66.33
SCIPY_VERSION = "1.15.3"
RECIPE_NAME = "flowers102_32_weakaug_300ep_v2"

FLOWERS_BASE_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
FLOWERS_FILES = {
    "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
    "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
    "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
}

LOCKED_PROTOCOL: Dict[str, Any] = {
    "dataset": "Oxford Flowers-102",
    "train_split": "official train+val (1020+1020=2040)",
    "evaluation_split": "official test (6149)",
    "model": "CIFAR-style ResNet56 (6n+2, n=9)",
    "image_size": IMAGE_SIZE,
    "num_classes": NUM_CLASSES,
    "planned_epochs": PLANNED_EPOCHS,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "test_batch_size": TEST_BATCH_SIZE,
    "optimizer": "SGD",
    "base_lr": BASE_LR,
    "momentum": MOMENTUM,
    "nesterov": True,
    "weight_decay": WEIGHT_DECAY,
    "weight_decay_exclusions": "bias and 1-D normalization parameters",
    "lr_schedule": "cosine to 0",
    "warmup_epochs": 0,
    "label_smoothing": 0.0,
    "mixup_alpha": 0.0,
    "cutmix_alpha": 0.0,
    "mixed_precision": False,
    "seed": SEED,
    "cudnn_benchmark": False,
    "train_drop_last": True,
    "strong_augmentation": False,
    "train_transform": "resize32+random_crop_padding4+hflip+normalize",
    "reference_top1": REFERENCE_TEACHER_TOP1,
    "protocol_basis": (
        "ALG explicit teacher constraints + attempt-1 300-epoch schedule + "
        "public LG weak-augmentation branch"
    ),
    "official_lg_commit": common.OFFICIAL_LG_COMMIT,
}


def log(message: str = "") -> None:
    common.log(message)


def ensure_scipy() -> None:
    try:
        scipy = importlib.import_module("scipy")
    except ImportError:
        log(f"[BOOT] scipy not found; installing scipy=={SCIPY_VERSION}")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                f"scipy=={SCIPY_VERSION}",
            ]
        )
        scipy = importlib.import_module("scipy")
        log("[BOOT] scipy installation completed")
    log(f"[BOOT] scipy={scipy.__version__}")


class ResNet56Flowers(common.ResNet56):
    """Official-LG-compatible ResNet56 with a 102-class classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.head.fc = nn.Linear(64, NUM_CLASSES, bias=True)
        self._init_weights(self.head.fc)


def flowers_train_transform() -> transforms.Compose:
    """Public-LG weak augmentation branch for the small Flowers split."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE), padding=IMAGE_SIZE // 8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(common.IMAGENET_MEAN, common.IMAGENET_STD),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the adjusted Flowers-102 ResNet56 teacher at 32x32"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--timing-run",
        action="store_true",
        help="Run two full-dataset epochs while retaining the 300-epoch LR schedule.",
    )
    modes.add_argument(
        "--smoke",
        action="store_true",
        help="Run one epoch on deterministic subsets.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke-train-samples", type=int, default=512)
    parser.add_argument("--smoke-test-samples", type=int, default=512)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.smoke_train_samples <= 0 or args.smoke_test_samples <= 0:
        raise ValueError("smoke subset sizes must be positive")


def actual_epochs(args: argparse.Namespace) -> int:
    if args.smoke:
        return 1
    if args.timing_run:
        return 2
    return PLANNED_EPOCHS


def default_run_name(args: argparse.Namespace) -> str:
    if args.smoke:
        suffix = "smoke"
    elif args.timing_run:
        suffix = "timing_2ep"
    else:
        suffix = "full_300ep"
    return f"teacher_resnet56_flowers102_32_weakaug_seed1_{suffix}"


def deterministic_subset(dataset: Dataset[Any], size: int, seed: int) -> Dataset[Any]:
    generator = torch.Generator().manual_seed(seed)
    count = min(size, len(dataset))
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    return Subset(dataset, indices)


def flowers_base(root: Path) -> Path:
    return root / "flowers-102"


def flowers_files_ready(root: Path) -> bool:
    base = flowers_base(root)
    image_dir = base / "jpg"
    if not image_dir.is_dir():
        return False
    if len(list(image_dir.glob("image_*.jpg"))) != 8189:
        return False
    for key in ("label", "setid"):
        filename, md5 = FLOWERS_FILES[key]
        if not check_integrity(str(base / filename), md5):
            return False
    return True


def download_file(
    url: str, destination: Path, expected_md5: str, source_name: str
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; IBAM-KD-H200-V2/1.0)",
            "Accept-Encoding": "identity",
        },
    )
    digest = hashlib.md5()
    downloaded = 0
    next_percent = 10
    next_bytes = 32 * 1024 * 1024
    log(f"[DATA] Download source={source_name} url={url}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open(
            "wb"
        ) as output:
            total = int(response.headers.get("Content-Length", "0"))
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = int(downloaded * 100 / total)
                    if percent >= next_percent:
                        log(
                            f"[DATA] Download progress source={source_name} "
                            f"{min(percent, 100)}% ({downloaded / 2**20:.1f} MiB)"
                        )
                        next_percent += 10
                elif downloaded >= next_bytes:
                    log(
                        f"[DATA] Download progress source={source_name} "
                        f"{downloaded / 2**20:.1f} MiB"
                    )
                    next_bytes += 32 * 1024 * 1024
        actual_md5 = digest.hexdigest()
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"MD5 mismatch: expected={expected_md5} actual={actual_md5}"
            )
        partial.replace(destination)
        log(
            f"[DATA] Download verified source={source_name} "
            f"size={downloaded / 2**20:.1f} MiB md5={actual_md5}"
        )
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def download_with_retry(
    url: str, destination: Path, expected_md5: str, source_name: str
) -> None:
    failures = []
    for attempt in range(1, 3):
        try:
            log(f"[DATA] Attempt source={source_name} try={attempt}/2")
            download_file(url, destination, expected_md5, source_name)
            return
        except Exception as error:
            destination.unlink(missing_ok=True)
            message = f"try={attempt}: {type(error).__name__}: {error}"
            failures.append(message)
            log(f"[DATA][WARN] source={source_name} {message}")
            if attempt < 2:
                time.sleep(3)
    raise RuntimeError(f"{source_name} download failed: {' | '.join(failures)}")


def ensure_flowers(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    base = flowers_base(root)
    base.mkdir(parents=True, exist_ok=True)
    if flowers_files_ready(root):
        log("[DATA] Existing Oxford Flowers files passed integrity checks")
        return

    image_archive, image_md5 = FLOWERS_FILES["image"]
    archive = base / image_archive
    if check_integrity(str(archive), image_md5):
        log(f"[DATA] Found verified image archive; extracting {archive}")
        extract_archive(str(archive), str(base))
    else:
        if archive.exists():
            log(f"[DATA][WARN] Removing incomplete or invalid archive: {archive}")
            archive.unlink()
        download_with_retry(
            FLOWERS_BASE_URL + image_archive,
            archive,
            image_md5,
            "Oxford official images",
        )
        log("[DATA] Extracting verified Oxford Flowers image archive")
        extract_archive(str(archive), str(base))

    for key, source_name in (
        ("label", "Oxford official labels"),
        ("setid", "Oxford official splits"),
    ):
        filename, md5 = FLOWERS_FILES[key]
        path = base / filename
        if check_integrity(str(path), md5):
            log(f"[DATA] Existing {filename} passed integrity check")
            continue
        if path.exists():
            log(f"[DATA][WARN] Removing incomplete or invalid file: {path}")
            path.unlink()
        download_with_retry(FLOWERS_BASE_URL + filename, path, md5, source_name)

    if not flowers_files_ready(root):
        raise RuntimeError("Oxford Flowers files failed integrity checks after download")
    log("[DATA] Oxford Flowers ready")


def build_datasets(args: argparse.Namespace) -> Tuple[Dataset[Any], Dataset[Any]]:
    ensure_scipy()
    log(f"[DATA] Oxford Flowers root={args.data_dir.resolve()}")
    log("[DATA] Preparing Oxford Flowers from official files with MD5 checks")
    ensure_flowers(args.data_dir)

    train_parts = [
        Flowers102(
            root=args.data_dir,
            split="train",
            transform=flowers_train_transform(),
            download=False,
        ),
        Flowers102(
            root=args.data_dir,
            split="val",
            transform=flowers_train_transform(),
            download=False,
        ),
    ]
    train_dataset: Dataset[Any] = ConcatDataset(train_parts)
    test_dataset: Dataset[Any] = Flowers102(
        root=args.data_dir,
        split="test",
        transform=common.official_test_transform(),
        download=False,
    )
    log(
        f"[DATA] Train split ready: official train+val "
        f"samples={len(train_parts[0])}+{len(train_parts[1])}={len(train_dataset)}"
    )
    log(f"[DATA] Eval split ready: official test samples={len(test_dataset)}")
    if args.smoke:
        train_dataset = deterministic_subset(
            train_dataset, args.smoke_train_samples, SEED
        )
        test_dataset = deterministic_subset(
            test_dataset, args.smoke_test_samples, SEED + 1
        )
    return train_dataset, test_dataset


def build_loaders(
    train_dataset: Dataset[Any], test_dataset: Dataset[Any], args: argparse.Namespace
) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    shared: Dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
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
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **shared,
    )
    log(f"[DATA] train_samples={len(train_dataset)} test_samples={len(test_dataset)}")
    log(
        f"[DATA] image_size={IMAGE_SIZE} train_batch={TRAIN_BATCH_SIZE} "
        f"test_batch={TEST_BATCH_SIZE} num_workers={args.num_workers} "
        "train_drop_last=True train_split=train+val eval_split=test"
    )
    return train_loader, test_loader


def protocol_check(model: nn.Module) -> None:
    problems = []
    expected_params = 861_750
    actual_params = common.count_parameters(model)
    if actual_params != expected_params:
        problems.append(f"parameter_count={actual_params} expected={expected_params}")
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        features = model.forward_features(torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE))
        logits = model.head(features[-1])
    model.train(was_training)
    expected_shapes = ((2, 16, 32, 32), (2, 32, 16, 16), (2, 64, 8, 8))
    actual_shapes = tuple(tuple(feature.shape) for feature in features)
    if actual_shapes != expected_shapes:
        problems.append(f"feature_shapes={actual_shapes} expected={expected_shapes}")
    if tuple(logits.shape) != (2, NUM_CLASSES):
        problems.append(f"logit_shape={tuple(logits.shape)} expected={(2, NUM_CLASSES)}")
    if problems:
        raise RuntimeError("Protocol self-check failed: " + "; ".join(problems))
    log(
        "[PROTOCOL_CHECK] status=PASS "
        f"params={actual_params:,} features={actual_shapes} logits={tuple(logits.shape)}"
    )


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    top1: float,
    best_top1: float,
    args: argparse.Namespace,
    epoch_times: Sequence[float],
) -> Dict[str, Any]:
    model_state = model.state_dict()
    return {
        "epoch": epoch,
        "accuracy": top1,
        "best_accuracy": best_top1,
        "test_err": 100.0 - top1,
        "ema_err": 100.0,
        "model_state": model_state,
        "model": model_state,
        "optimizer_state": optimizer.state_dict(),
        "model_name": "ResNet56",
        "architecture": "CIFAR-style ResNet56 (6n+2, n=9)",
        "dataset": "Oxford Flowers-102",
        "train_split": "train+val",
        "evaluation_split": "test",
        "num_classes": NUM_CLASSES,
        "input_resolution": IMAGE_SIZE,
        "reference_teacher_top1": REFERENCE_TEACHER_TOP1,
        "recipe_name": RECIPE_NAME,
        "official_lg_commit": common.OFFICIAL_LG_COMMIT,
        "protocol": LOCKED_PROTOCOL,
        "preprocessing": {
            "normalization_mean": common.IMAGENET_MEAN,
            "normalization_std": common.IMAGENET_STD,
            "strong_augmentation": False,
            "train_transform": LOCKED_PROTOCOL["train_transform"],
            "timm_version": common.timm.__version__,
        },
        "epoch_times": list(epoch_times),
        "mode": "smoke" if args.smoke else "timing" if args.timing_run else "full",
    }


def write_metrics_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                "epoch",
                "train_loss",
                "train_top1",
                "test_top1",
                "best_top1",
                "closest_to_reference_top1",
                "learning_rate",
                "epoch_seconds",
                "elapsed_seconds",
            ]
        )


def append_metric(path: Path, row: Sequence[Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(row)


def train(args: argparse.Namespace) -> None:
    common.install_signal_handlers()
    validate_args(args)
    common.seed_everything(SEED)
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = False
    epochs_to_run = actual_epochs(args)
    run_name = args.run_name or default_run_name(args)
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "teacher_resnet56_flowers102_32_weakaug_best.pt"
    latest_path = run_dir / "teacher_resnet56_flowers102_32_weakaug_latest.pt"
    closest_path = (
        run_dir / "teacher_resnet56_flowers102_32_weakaug_closest_to_reference.pt"
    )
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"

    log("=" * 80)
    log("TRAIN FLOWERS-102 RESNET56 TEACHER RECIPE V2 (32 x 32)")
    log("=" * 80)
    log(f"[ENV] python={sys.version.split()[0]} torch={torch.__version__}")
    log(
        f"[ENV] torchvision={common.torchvision.__version__} "
        f"timm={common.timm.__version__}"
    )
    log(
        f"[ENV] cuda_available={torch.cuda.is_available()} "
        f"cuda_device_count={torch.cuda.device_count()}"
    )
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        log(f"[ENV] gpu_name={torch.cuda.get_device_name(0)}")
        log(f"[ENV] gpu_memory_gib={properties.total_memory / 2**30:.2f}")
    log(f"[ENV] device={device} amp={amp_enabled} seed={SEED}")
    log(f"[PATH] data_dir={args.data_dir.resolve()}")
    log(f"[PATH] run_dir={run_dir.resolve()}")
    log(f"[PATH] best_checkpoint={best_path.resolve()}")
    log(
        f"[MODE] smoke={args.smoke} timing_run={args.timing_run} "
        f"actual_epochs={epochs_to_run} planned_epochs={PLANNED_EPOCHS}"
    )
    log(
        f"[REFERENCE] flowers_teacher_top1={REFERENCE_TEACHER_TOP1:.2f}% "
        f"lg_commit={common.OFFICIAL_LG_COMMIT}"
    )
    log(f"[RECIPE] name={RECIPE_NAME}")
    log(
        "[NOTE] Flowers teacher epochs and augmentation are not published in "
        "the available Flowers YAML; this is a documented controlled adjustment."
    )

    model = ResNet56Flowers()
    protocol_check(model)
    model = model.to(device)
    train_dataset, test_dataset = build_datasets(args)
    train_loader, test_loader = build_loaders(train_dataset, test_dataset, args)

    optimizer = torch.optim.SGD(
        common.parameter_groups(model),
        lr=BASE_LR,
        momentum=MOMENTUM,
        dampening=0.0,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PLANNED_EPOCHS, eta_min=0.0
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    scaler = common.create_grad_scaler(amp_enabled)

    config_payload = {
        "protocol": LOCKED_PROTOCOL,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": common.torchvision.__version__,
            "timm": common.timm.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "amp": amp_enabled,
            "num_workers": args.num_workers,
            "actual_epochs": epochs_to_run,
        },
    }
    common.atomic_json_save(config_payload, config_path)
    write_metrics_header(metrics_path)

    log(
        "[PROTOCOL] input=32 optimizer=SGD lr=0.1 momentum=0.9 "
        "nesterov=True weight_decay=0.0005 warmup=0 cosine_to=0"
    )
    log(
        "[AUG] official_weak_branch=True resize=32 random_crop_padding=4 "
        "horizontal_flip=0.5 normalization=ImageNet"
    )
    log(
        "[AUG] removed_from_v1=random_resized_crop+randaugment+random_erasing"
    )
    log(
        f"[MODEL] teacher_params={common.count_parameters(model):,} "
        "architecture=CIFAR-ResNet56 classes=102"
    )

    best_top1 = -1.0
    latest_top1 = -1.0
    closest_top1 = -1.0
    closest_distance = float("inf")
    epoch_times: list[float] = []
    start_time = time.time()
    last_epoch = 0

    for epoch in range(1, epochs_to_run + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with common.autocast_context(amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.size(0)
            loss_sum += float(loss.detach().item()) * batch_size
            correct += common.top1_correct(logits.detach(), targets)
            total += batch_size

        latest_top1 = common.evaluate(model, test_loader, device, amp_enabled)
        scheduler.step()
        epoch_seconds = time.time() - epoch_start
        epoch_times.append(epoch_seconds)
        elapsed_seconds = time.time() - start_time
        average_epoch = sum(epoch_times) / len(epoch_times)
        estimated_full = average_epoch * PLANNED_EPOCHS
        train_loss = loss_sum / max(total, 1)
        train_top1 = 100.0 * correct / max(total, 1)
        last_epoch = epoch

        is_best = latest_top1 > best_top1
        if is_best:
            best_top1 = latest_top1
        distance = abs(latest_top1 - REFERENCE_TEACHER_TOP1)
        is_closest = distance < closest_distance
        if is_closest:
            closest_distance = distance
            closest_top1 = latest_top1

        payload = checkpoint_payload(
            model, optimizer, epoch, latest_top1, best_top1, args, epoch_times
        )
        common.atomic_torch_save(payload, latest_path)
        if is_best:
            common.atomic_torch_save(payload, best_path)
        if is_closest:
            common.atomic_torch_save(payload, closest_path)

        append_metric(
            metrics_path,
            [
                epoch,
                f"{train_loss:.8f}",
                f"{train_top1:.6f}",
                f"{latest_top1:.6f}",
                f"{best_top1:.6f}",
                f"{closest_top1:.6f}",
                f"{current_lr:.10f}",
                f"{epoch_seconds:.6f}",
                f"{elapsed_seconds:.6f}",
            ],
        )

        partial_summary = {
            "status": "running",
            "completed_epoch": epoch,
            "planned_epochs": PLANNED_EPOCHS,
            "actual_epochs_this_run": epochs_to_run,
            "latest_top1": latest_top1,
            "best_top1": best_top1,
            "closest_to_reference_top1": closest_top1,
            "reference_top1": REFERENCE_TEACHER_TOP1,
            "avg_epoch_seconds": average_epoch,
            "estimated_planned_seconds": estimated_full,
            "estimated_planned_human": common.format_duration(estimated_full),
            "elapsed_seconds": elapsed_seconds,
            "paths": {
                "best": str(best_path.resolve()),
                "latest": str(latest_path.resolve()),
                "closest_to_reference": str(closest_path.resolve()),
            },
            "protocol": LOCKED_PROTOCOL,
        }
        common.atomic_json_save(partial_summary, summary_path)

        log(
            f"[TEACHER][{epoch:03d}/{epochs_to_run:03d}] "
            f"loss={train_loss:.4f} train_acc={train_top1:.2f}% "
            f"test_acc={latest_top1:.2f}% best={best_top1:.2f}% "
            f"closest_ref={closest_top1:.2f}% lr={current_lr:.8f} "
            f"time={epoch_seconds:.1f}s avg_epoch={average_epoch:.1f}s "
            f"est_{PLANNED_EPOCHS}={common.format_duration(estimated_full)} "
            f"elapsed={common.format_duration(elapsed_seconds)}"
            + (" saved_best" if is_best else "")
            + (" saved_closest" if is_closest else "")
        )

    total_elapsed = time.time() - start_time
    average_epoch = sum(epoch_times) / len(epoch_times)
    estimated_full = average_epoch * PLANNED_EPOCHS
    hashes = {
        "best": common.sha256_file(best_path),
        "latest": common.sha256_file(latest_path),
        "closest_to_reference": common.sha256_file(closest_path),
    }
    final_summary = {
        "status": "completed",
        "mode": "smoke" if args.smoke else "timing" if args.timing_run else "full",
        "completed_epoch": last_epoch,
        "planned_epochs": PLANNED_EPOCHS,
        "latest_top1": latest_top1,
        "best_top1": best_top1,
        "closest_to_reference_top1": closest_top1,
        "reference_top1": REFERENCE_TEACHER_TOP1,
        "best_gap_to_reference_pp": best_top1 - REFERENCE_TEACHER_TOP1,
        "closest_gap_to_reference_pp": closest_top1 - REFERENCE_TEACHER_TOP1,
        "avg_epoch_seconds": average_epoch,
        "estimated_planned_seconds": estimated_full,
        "estimated_planned_human": common.format_duration(estimated_full),
        "elapsed_seconds": total_elapsed,
        "elapsed_human": common.format_duration(total_elapsed),
        "paths": {
            "best": str(best_path.resolve()),
            "latest": str(latest_path.resolve()),
            "closest_to_reference": str(closest_path.resolve()),
            "config": str(config_path.resolve()),
            "metrics": str(metrics_path.resolve()),
        },
        "sha256": hashes,
        "protocol": LOCKED_PROTOCOL,
    }
    common.atomic_json_save(final_summary, summary_path)

    log("=" * 80)
    log(
        f"[FINAL_RESULT] teacher_best_top1={best_top1:.2f}% "
        f"reference_teacher_top1={REFERENCE_TEACHER_TOP1:.2f}% "
        f"gap_to_reference={best_top1 - REFERENCE_TEACHER_TOP1:+.2f}pp"
    )
    log(
        f"[FINAL_RESULT] closest_to_reference_top1={closest_top1:.2f}% "
        f"gap={closest_top1 - REFERENCE_TEACHER_TOP1:+.2f}pp"
    )
    log(
        f"[TIMING] teacher_avg_epoch={average_epoch:.1f}s "
        f"estimated_{PLANNED_EPOCHS}_teacher={common.format_duration(estimated_full)} "
        f"elapsed={common.format_duration(total_elapsed)}"
    )
    log(f"[FINAL_RESULT] best_checkpoint={best_path.resolve()}")
    log(f"[FINAL_RESULT] latest_checkpoint={latest_path.resolve()}")
    log(f"[FINAL_RESULT] closest_checkpoint={closest_path.resolve()}")
    log(f"[FINAL_RESULT] summary={summary_path.resolve()}")
    log(f"[SHA256] best={hashes['best']}")
    log(f"[SHA256] latest={hashes['latest']}")
    log(f"[SHA256] closest={hashes['closest_to_reference']}")
    log("[DONE] Flowers teacher training completed successfully; resources may be released.")


def main() -> None:
    try:
        train(parse_args())
    except Exception as error:
        log("=" * 80)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Flowers teacher training did not complete.")
        raise


if __name__ == "__main__":
    main()
