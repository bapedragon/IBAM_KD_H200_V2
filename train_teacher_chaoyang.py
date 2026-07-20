#!/usr/bin/env python3
"""Train the Chaoyang ResNet56 guidance teacher at 32 x 32.

The dataset must be mounted at ``/app/data/chaoyang`` and contain the official
train/test folders plus ``train.json`` and ``test.json``. The statistical
recipe is locked; only paths, run name, worker count, and smoke/timing modes
are operational command-line options.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import train_teacher_cifar100 as common


NUM_CLASSES = 4
IMAGE_SIZE = 32
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 200
PLANNED_EPOCHS = 300
BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
SEED = 1
REFERENCE_TEACHER_TOP1 = 77.20
RECIPE_NAME = "chaoyang_32_moderateaug_300ep_v1"
CLASS_NAMES = {
    0: "normal",
    1: "serrated",
    2: "adenocarcinoma",
    3: "adenoma",
}
EXPECTED_COUNTS = {
    "train": {0: 1111, 1: 842, 2: 1404, 3: 664},
    "test": {0: 705, 1: 321, 2: 840, 3: 273},
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

LOCKED_PROTOCOL: Dict[str, Any] = {
    "dataset": "Chaoyang",
    "train_split": "official train (4021)",
    "evaluation_split": "official test (2139)",
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
    "train_transform": "random_resized_crop32_scale0.8-1.0_bicubic+hflip",
    "normalization": "ImageNet mean/std",
    "reference_top1": REFERENCE_TEACHER_TOP1,
    "protocol_basis": (
        "ALG explicit teacher constraints + previously validated Chaoyang "
        "moderate crop policy; exact Chaoyang teacher YAML unavailable"
    ),
    "official_lg_commit": common.OFFICIAL_LG_COMMIT,
}


def log(message: str = "") -> None:
    common.log(message)


class ResNet56Chaoyang(common.ResNet56):
    def __init__(self) -> None:
        super().__init__()
        self.head.fc = nn.Linear(64, NUM_CLASSES, bias=True)
        self._init_weights(self.head.fc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Chaoyang ResNet56 guidance teacher at 32x32"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data/chaoyang"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--timing-run",
        action="store_true",
        help="Run two full-data epochs while retaining the 300-epoch LR schedule.",
    )
    modes.add_argument("--smoke", action="store_true")
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
    return f"teacher_resnet56_chaoyang_32_moderateaug_seed1_{suffix}"


def is_dataset_root(path: Path) -> bool:
    return all(
        (
            (path / "train").is_dir(),
            (path / "test").is_dir(),
            (path / "train.json").is_file(),
            (path / "test.json").is_file(),
        )
    )


def resolve_dataset_root(requested: Path) -> Path:
    requested = requested.expanduser().resolve()
    if not requested.exists():
        raise FileNotFoundError(f"Mounted Chaoyang path does not exist: {requested}")
    if not requested.is_dir():
        raise NotADirectoryError(f"Chaoyang --data-dir is not a directory: {requested}")
    if is_dataset_root(requested):
        return requested

    candidates: list[Path] = []
    for metadata in requested.rglob("train.json"):
        if "__MACOSX" in metadata.parts:
            continue
        parent = metadata.parent
        try:
            depth = len(parent.relative_to(requested).parts)
        except ValueError:
            continue
        if depth <= 3 and is_dataset_root(parent):
            candidates.append(parent.resolve())
    unique = sorted(set(candidates))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise FileNotFoundError(
            "Could not find train/, test/, train.json, and test.json under "
            f"{requested} (searched up to three nested levels)"
        )
    raise RuntimeError(
        "Multiple Chaoyang dataset roots found: " + ", ".join(map(str, unique))
    )


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(common.IMAGENET_MEAN, common.IMAGENET_STD),
        ]
    )


def test_transform() -> transforms.Compose:
    return common.official_test_transform()


class ChaoyangDataset(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(self, root: Path, split: str, transform: Any) -> None:
        if split not in EXPECTED_COUNTS:
            raise ValueError(f"Unsupported Chaoyang split: {split}")
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = self._load_and_validate_records()

    def _load_and_validate_records(self) -> list[Tuple[Path, int]]:
        metadata_path = self.root / f"{self.split}.json"
        try:
            records = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid JSON metadata: {metadata_path}: {error}") from error
        if not isinstance(records, list):
            raise TypeError(f"Expected a JSON list in {metadata_path}")

        samples: list[Tuple[Path, int]] = []
        seen_names: set[str] = set()
        for index, record in enumerate(records):
            if not isinstance(record, dict) or "name" not in record or "label" not in record:
                raise ValueError(f"Invalid record at {metadata_path}[{index}]: {record!r}")
            relative_name = str(record["name"])
            label = int(record["label"])
            if label not in CLASS_NAMES:
                raise ValueError(f"Invalid label={label} at {metadata_path}[{index}]")
            if relative_name in seen_names:
                raise ValueError(f"Duplicate image record in {metadata_path}: {relative_name}")
            seen_names.add(relative_name)
            image_path = self.root / relative_name
            if not image_path.is_file():
                raise FileNotFoundError(f"Image listed in JSON is missing: {image_path}")
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported image extension: {image_path}")
            samples.append((image_path, label))

        actual_counts = dict(sorted(Counter(label for _, label in samples).items()))
        expected_counts = EXPECTED_COUNTS[self.split]
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"Unexpected {self.split} class counts: expected={expected_counts} "
                f"actual={actual_counts}"
            )
        folder_images = {
            path.resolve()
            for path in (self.root / self.split).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        listed_images = {path.resolve() for path, _ in samples}
        if folder_images != listed_images:
            raise RuntimeError(
                f"{self.split} image/JSON mismatch: "
                f"unlisted_images={len(folder_images - listed_images)} "
                f"missing_images={len(listed_images - folder_images)}"
            )
        with Image.open(samples[0][0]) as image:
            first_size = image.size
            image.verify()
        if first_size != (512, 512):
            raise RuntimeError(f"Unexpected Chaoyang image size: {first_size}")
        rendered = " ".join(
            f"{CLASS_NAMES[label]}={actual_counts[label]}" for label in range(NUM_CLASSES)
        )
        log(
            f"[DATA] {self.split}_split verified samples={len(samples)} "
            f"{rendered} first_image_size={first_size}"
        )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, label


def deterministic_subset(dataset: Dataset[Any], size: int, seed: int) -> Dataset[Any]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[: min(size, len(dataset))]
    return Subset(dataset, indices.tolist())


def build_datasets(args: argparse.Namespace) -> Tuple[Dataset[Any], Dataset[Any], Path]:
    root = resolve_dataset_root(args.data_dir)
    log(f"[DATA] requested_root={args.data_dir.expanduser().resolve()}")
    log(f"[DATA] resolved_root={root}")
    log("[DATA] Validating official Chaoyang train/test splits and labels")
    train_dataset: Dataset[Any] = ChaoyangDataset(root, "train", train_transform())
    test_dataset: Dataset[Any] = ChaoyangDataset(root, "test", test_transform())
    if args.smoke:
        train_dataset = deterministic_subset(train_dataset, args.smoke_train_samples, SEED)
        test_dataset = deterministic_subset(test_dataset, args.smoke_test_samples, SEED + 1)
    return train_dataset, test_dataset, root


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
        generator=torch.Generator().manual_seed(SEED),
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
        "train_drop_last=True"
    )
    return train_loader, test_loader


def protocol_check(model: nn.Module) -> None:
    expected_params = 855_380
    actual_params = common.count_parameters(model)
    with torch.inference_mode():
        features = model.forward_features(torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE))
        logits = model(torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE))
    shapes = [tuple(feature.shape[-2:]) for feature in features]
    if actual_params != expected_params:
        raise RuntimeError(f"ResNet56 parameter mismatch: {actual_params} != {expected_params}")
    if shapes != [(32, 32), (16, 16), (8, 8)] or tuple(logits.shape) != (2, 4):
        raise RuntimeError(f"Unexpected feature/logit shapes: {shapes}, {tuple(logits.shape)}")
    log(
        f"[PROTOCOL_CHECK] status=PASS params={actual_params:,} "
        f"features={shapes} logits={tuple(logits.shape)}"
    )


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    top1: float,
    best_top1: float,
    root: Path,
    args: argparse.Namespace,
    epoch_times: Sequence[float],
) -> Dict[str, Any]:
    state = model.state_dict()
    return {
        "epoch": epoch,
        "accuracy": top1,
        "best_accuracy": best_top1,
        "test_err": 100.0 - top1,
        "model_state": state,
        "model": state,
        "optimizer_state": optimizer.state_dict(),
        "model_name": "ResNet56",
        "architecture": "CIFAR-style ResNet56 (6n+2, n=9)",
        "dataset": "Chaoyang",
        "dataset_root": str(root),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "input_resolution": IMAGE_SIZE,
        "reference_teacher_top1": REFERENCE_TEACHER_TOP1,
        "recipe_name": RECIPE_NAME,
        "official_lg_commit": common.OFFICIAL_LG_COMMIT,
        "protocol": LOCKED_PROTOCOL,
        "preprocessing": {
            "normalization_mean": common.IMAGENET_MEAN,
            "normalization_std": common.IMAGENET_STD,
            "train_transform": LOCKED_PROTOCOL["train_transform"],
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
                "closest_top1",
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
    best_path = run_dir / "teacher_resnet56_chaoyang_32_best.pt"
    latest_path = run_dir / "teacher_resnet56_chaoyang_32_latest.pt"
    closest_path = run_dir / "teacher_resnet56_chaoyang_32_closest_to_reference.pt"
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"

    log("=" * 80)
    log("TRAIN CHAOYANG RESNET56 GUIDANCE TEACHER (32 x 32)")
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
    log(f"[PATH] requested_data_dir={args.data_dir.expanduser().resolve()}")
    log(f"[PATH] run_dir={run_dir.resolve()}")
    log(f"[PATH] best_checkpoint={best_path.resolve()}")
    log(
        f"[MODE] smoke={args.smoke} timing_run={args.timing_run} "
        f"actual_epochs={epochs_to_run} planned_epochs={PLANNED_EPOCHS}"
    )
    log(
        f"[REFERENCE] chaoyang_teacher_top1={REFERENCE_TEACHER_TOP1:.2f}% "
        f"lg_commit={common.OFFICIAL_LG_COMMIT}"
    )
    log(f"[RECIPE] name={RECIPE_NAME}")
    log(
        "[NOTE] The exact Chaoyang teacher YAML is unavailable; the moderate "
        "crop policy is a documented dataset-specific implementation choice."
    )

    model = ResNet56Chaoyang()
    protocol_check(model)
    model = model.to(device)
    train_dataset, test_dataset, dataset_root = build_datasets(args)
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

    common.atomic_json_save(
        {
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
                "resolved_data_root": str(dataset_root),
            },
        },
        config_path,
    )
    write_metrics_header(metrics_path)
    log(
        "[PROTOCOL] input=32 optimizer=SGD lr=0.1 momentum=0.9 "
        "nesterov=True weight_decay=0.0005 warmup=0 cosine_to=0"
    )
    log(
        "[AUG] random_resized_crop=32 scale=(0.8,1.0) interpolation=bicubic "
        "horizontal_flip=0.5 normalization=ImageNet"
    )
    log(
        f"[MODEL] teacher_params={common.count_parameters(model):,} "
        "architecture=CIFAR-ResNet56 classes=4"
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
            count = targets.size(0)
            loss_sum += float(loss.detach().item()) * count
            correct += common.top1_correct(logits.detach(), targets)
            total += count

        latest_top1 = common.evaluate(model, test_loader, device, amp_enabled)
        scheduler.step()
        epoch_seconds = time.time() - epoch_start
        epoch_times.append(epoch_seconds)
        elapsed = time.time() - start_time
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
            model,
            optimizer,
            epoch,
            latest_top1,
            best_top1,
            dataset_root,
            args,
            epoch_times,
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
                f"{elapsed:.6f}",
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
            "estimated_300_seconds": estimated_full,
            "estimated_300_human": common.format_duration(estimated_full),
            "elapsed_seconds": elapsed,
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
            f"est_300={common.format_duration(estimated_full)} "
            f"elapsed={common.format_duration(elapsed)}"
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
        "estimated_300_seconds": estimated_full,
        "estimated_300_human": common.format_duration(estimated_full),
        "elapsed_seconds": total_elapsed,
        "elapsed_human": common.format_duration(total_elapsed),
        "resolved_data_root": str(dataset_root),
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
        f"estimated_300_teacher={common.format_duration(estimated_full)} "
        f"elapsed={common.format_duration(total_elapsed)}"
    )
    log(f"[FINAL_RESULT] best_checkpoint={best_path.resolve()}")
    log(f"[FINAL_RESULT] latest_checkpoint={latest_path.resolve()}")
    log(f"[FINAL_RESULT] closest_checkpoint={closest_path.resolve()}")
    log(f"[FINAL_RESULT] summary={summary_path.resolve()}")
    log(f"[SHA256] best={hashes['best']}")
    log(f"[SHA256] latest={hashes['latest']}")
    log(f"[SHA256] closest={hashes['closest_to_reference']}")
    log("[DONE] Chaoyang teacher training completed successfully; resources may be released.")


def main() -> None:
    try:
        train(parse_args())
    except Exception as error:
        log("=" * 80)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Chaoyang teacher training did not complete.")
        raise


if __name__ == "__main__":
    main()
