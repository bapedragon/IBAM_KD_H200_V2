#!/usr/bin/env python3
"""Train the LG-compatible CIFAR-100 ResNet56 guidance teacher at 32 x 32.

The locked full-run defaults reproduce the official LG ResNet56 configuration.
On the KAU H200 runner, pass ``--output-dir /app/output`` so artifacts are
collected after the Pod is released.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple


TIMM_VERSION = "1.0.27"


def log(message: str = "") -> None:
    print(message, flush=True)


def ensure_timm() -> Any:
    try:
        timm_module = importlib.import_module("timm")
    except ImportError:
        log(f"[BOOT] timm not found; installing timm=={TIMM_VERSION}")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                f"timm=={TIMM_VERSION}",
            ]
        )
        timm_module = importlib.import_module("timm")
        log("[BOOT] timm installation completed")
    if timm_module.__version__ != TIMM_VERSION:
        raise RuntimeError(
            f"Expected timm=={TIMM_VERSION}, found timm=={timm_module.__version__}"
        )
    return timm_module


log(f"[BOOT] Python process started: {sys.executable}")
timm = ensure_timm()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from timm.data import create_transform
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.datasets.utils import check_integrity, extract_archive


OFFICIAL_LG_COMMIT = "d2165f74049c906b0afc9f957491960fb3c0cc8b"
OFFICIAL_LG_CONFIG = "configs/resnet/r-56_c100.yaml"
REFERENCE_TEACHER_TOP1 = 70.43
NUM_CLASSES = 100
IMAGE_SIZE = 32
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 200
PLANNED_EPOCHS = 300
BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
SEED = 1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CIFAR100_SOURCES = (
    (
        "Hugging Face mirror",
        "https://huggingface.co/datasets/nakroy/cifar100-python/resolve/"
        "201a32345d2c6b970e1a36c582930c83e09c96d2/cifar-100-python.tar.gz",
    ),
    (
        "SJTU mirror",
        "https://scidata.sjtu.edu.cn/records/xk2s3-v1e12/files/"
        "cifar-100-python.tar.gz?download=1",
    ),
    ("Toronto official", CIFAR100.url),
)

LOCKED_PROTOCOL: Dict[str, Any] = {
    "dataset": "CIFAR-100",
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
    "amp_default": False,
    "seed": SEED,
    "cudnn_benchmark": False,
    "train_drop_last": True,
    "reference_top1": REFERENCE_TEACHER_TOP1,
    "official_lg_commit": OFFICIAL_LG_COMMIT,
    "official_lg_config": OFFICIAL_LG_CONFIG,
}


def install_signal_handlers() -> None:
    def handle_signal(signum: int, frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        log("=" * 80)
        log(f"[FATAL][SIGNAL] Received {signal_name}; external termination requested.")
        if frame is not None:
            traceback.print_stack(frame)
        log("[FATAL] The last atomically completed epoch remains on disk.")
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_signal)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResStemCifar(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.af = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.af(self.bn(self.conv(x)))


class BasicTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.a = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.a_bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.a_af = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.b_bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_af(x)
        x = self.b(x)
        return self.b_bn(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        if in_channels != out_channels or stride != 1:
            self.proj: nn.Module | None = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            self.bn: nn.Module | None = nn.BatchNorm2d(
                out_channels, eps=1e-5, momentum=0.1
            )
        else:
            self.proj = None
            self.bn = None
        self.f = BasicTransform(in_channels, out_channels, stride)
        self.af = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.bn(self.proj(x)) if self.proj is not None else x
        return self.af(residual + self.f(x))


class ResStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, depth: int) -> None:
        super().__init__()
        for index in range(depth):
            block_stride = stride if index == 0 else 1
            block_in = in_channels if index == 0 else out_channels
            self.add_module(
                f"b{index + 1}", ResBlock(block_in, out_channels, block_stride)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.children():
            x = block(x)
        return x


class ResHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, NUM_CLASSES, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        return self.fc(torch.flatten(x, 1))


class ResNet56(nn.Module):
    """Official-LG-compatible CIFAR ResNet56 with 861,620 parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = ResStemCifar()
        self.s1 = ResStage(16, 16, stride=1, depth=9)
        self.s2 = ResStage(16, 32, stride=2, depth=9)
        self.s3 = ResStage(32, 64, stride=2, depth=9)
        self.head = ResHead()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        return f1, f2, f3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x)[-1])


class OfficialLGTrainTransform:
    def __init__(self) -> None:
        primary, secondary, final = create_transform(
            input_size=(IMAGE_SIZE, IMAGE_SIZE),
            is_training=True,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5-inc1",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            interpolation="bicubic",
            separate=True,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
        self.primary = primary
        self.secondary = secondary
        self.final = final

    def __call__(self, image: Any) -> torch.Tensor:
        image = self.primary(image)
        image = self.secondary(image)
        return self.final(image)

    def __repr__(self) -> str:
        return (
            "OfficialLGTrainTransform("
            f"primary={self.primary}, secondary={self.secondary}, final={self.final})"
        )


def official_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the LG-compatible CIFAR-100 ResNet56 teacher at 32x32"
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
    parser.add_argument("--smoke-train-samples", type=int, default=1024)
    parser.add_argument("--smoke-test-samples", type=int, default=512)
    parser.add_argument(
        "--amp",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Optional operational override. Official LG teacher training uses FP32.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.smoke_train_samples <= 0 or args.smoke_test_samples <= 0:
        raise ValueError("smoke subset sizes must be positive")
    if args.amp:
        log("[PROTOCOL][WARN] AMP is enabled; the official LG configuration uses FP32.")


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
    return f"teacher_resnet56_cifar100_32_lg_official_seed1_{suffix}"


def deterministic_subset(dataset: Dataset[Any], size: int, seed: int) -> Dataset[Any]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[: min(size, len(dataset))]
    return Subset(dataset, indices.tolist())


def cifar100_files_ready(root: Path) -> bool:
    base = root / CIFAR100.base_folder
    required = list(CIFAR100.train_list) + list(CIFAR100.test_list)
    required.append((CIFAR100.meta["filename"], CIFAR100.meta["md5"]))
    return all(check_integrity(str(base / name), md5) for name, md5 in required)


def download_archive(url: str, destination: Path, source_name: str) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
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
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
            total = int(response.headers.get("Content-Length", "0"))
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
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
        if actual_md5 != CIFAR100.tgz_md5:
            raise RuntimeError(
                f"MD5 mismatch: expected={CIFAR100.tgz_md5} actual={actual_md5}"
            )
        partial.replace(destination)
        log(
            f"[DATA] Download verified source={source_name} "
            f"size={downloaded / 2**20:.1f} MiB md5={actual_md5}"
        )
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def ensure_cifar100(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if cifar100_files_ready(root):
        log("[DATA] Existing CIFAR-100 files passed integrity checks")
        return
    archive = root / CIFAR100.filename
    if check_integrity(str(archive), CIFAR100.tgz_md5):
        log(f"[DATA] Found verified archive; extracting {archive}")
        extract_archive(str(archive), str(root))
        if cifar100_files_ready(root):
            log("[DATA] CIFAR-100 extraction and integrity checks completed")
            return
    elif archive.exists():
        log(f"[DATA][WARN] Removing incomplete or invalid archive: {archive}")
        archive.unlink()

    failures = []
    for source_name, url in CIFAR100_SOURCES:
        for attempt in range(1, 3):
            try:
                log(f"[DATA] Attempt source={source_name} try={attempt}/2")
                download_archive(url, archive, source_name)
                log(f"[DATA] Extracting verified archive from {source_name}")
                extract_archive(str(archive), str(root))
                if not cifar100_files_ready(root):
                    raise RuntimeError("extracted CIFAR-100 files failed integrity checks")
                log(f"[DATA] CIFAR-100 ready from {source_name}")
                return
            except Exception as error:
                message = f"{source_name} try={attempt}: {type(error).__name__}: {error}"
                failures.append(message)
                log(f"[DATA][WARN] {message}")
                archive.unlink(missing_ok=True)
                if attempt < 2:
                    time.sleep(3)
    raise RuntimeError("All CIFAR-100 sources failed: " + " | ".join(failures))


def build_datasets(args: argparse.Namespace) -> Tuple[Dataset[Any], Dataset[Any]]:
    log(f"[DATA] CIFAR-100 root={args.data_dir.resolve()}")
    log("[DATA] Preparing verified CIFAR-100 files")
    ensure_cifar100(args.data_dir)
    train_dataset: Dataset[Any] = CIFAR100(
        root=args.data_dir,
        train=True,
        transform=OfficialLGTrainTransform(),
        download=False,
    )
    test_dataset: Dataset[Any] = CIFAR100(
        root=args.data_dir,
        train=False,
        transform=official_test_transform(),
        download=False,
    )
    if args.smoke:
        train_dataset = deterministic_subset(train_dataset, args.smoke_train_samples, SEED)
        test_dataset = deterministic_subset(test_dataset, args.smoke_test_samples, SEED + 1)
    return train_dataset, test_dataset


def build_loaders(
    train_dataset: Dataset[Any], test_dataset: Dataset[Any], args: argparse.Namespace
) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    common: Dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **common,
    )
    log(f"[DATA] train_samples={len(train_dataset)} test_samples={len(test_dataset)}")
    log(
        f"[DATA] image_size={IMAGE_SIZE} train_batch={TRAIN_BATCH_SIZE} "
        f"test_batch={TEST_BATCH_SIZE} num_workers={args.num_workers} "
        f"train_drop_last=True"
    )
    return train_loader, test_loader


def parameter_groups(model: nn.Module) -> Sequence[Dict[str, Any]]:
    decay, no_decay = [], []
    decay_names, no_decay_names = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias"):
            no_decay.append(parameter)
            no_decay_names.append(name)
        else:
            decay.append(parameter)
            decay_names.append(name)
    if len(decay_names) + len(no_decay_names) != len(list(model.parameters())):
        raise RuntimeError("Optimizer parameter grouping is incomplete")
    log(
        f"[OPTIM] weight_decay_params={sum(p.numel() for p in decay):,} "
        f"no_decay_params={sum(p.numel() for p in no_decay):,}"
    )
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": WEIGHT_DECAY},
    ]


def create_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def top1_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    return int(logits.argmax(dim=1).eq(targets).sum().item())


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: Iterable[Any], device: torch.device, amp_enabled: bool
) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(amp_enabled):
            logits = model(images)
        correct += top1_correct(logits, targets)
        total += targets.size(0)
    return 100.0 * correct / max(total, 1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: Dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def atomic_json_save(payload: Dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, destination)


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
        "dataset": "CIFAR-100",
        "num_classes": NUM_CLASSES,
        "input_resolution": IMAGE_SIZE,
        "reference_teacher_top1": REFERENCE_TEACHER_TOP1,
        "official_lg_commit": OFFICIAL_LG_COMMIT,
        "official_lg_config": OFFICIAL_LG_CONFIG,
        "protocol": LOCKED_PROTOCOL,
        "preprocessing": {
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
            "strong_augmentation": True,
            "timm_version": timm.__version__,
        },
        "epoch_times": list(epoch_times),
        "mode": "smoke" if args.smoke else "timing" if args.timing_run else "full",
    }


def write_metrics_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
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


def protocol_check(model: nn.Module) -> None:
    problems = []
    if count_parameters(model) != 861_620:
        problems.append(f"parameter_count={count_parameters(model)} expected=861620")
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
        f"params={count_parameters(model):,} features={actual_shapes} logits={tuple(logits.shape)}"
    )


def train(args: argparse.Namespace) -> None:
    install_signal_handlers()
    validate_args(args)
    seed_everything(SEED)
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    epochs_to_run = actual_epochs(args)
    run_name = args.run_name or default_run_name(args)
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "teacher_resnet56_cifar100_32_best.pt"
    latest_path = run_dir / "teacher_resnet56_cifar100_32_latest.pt"
    closest_path = run_dir / "teacher_resnet56_cifar100_32_closest_to_lg_reference.pt"
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"

    log("=" * 80)
    log("TRAIN LG-COMPATIBLE CIFAR-100 RESNET56 TEACHER (32 x 32)")
    log("=" * 80)
    log(f"[ENV] python={sys.version.split()[0]} torch={torch.__version__}")
    log(f"[ENV] torchvision={torchvision.__version__} timm={timm.__version__}")
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
        f"[REFERENCE] lg_teacher_top1={REFERENCE_TEACHER_TOP1:.2f}% "
        f"official_commit={OFFICIAL_LG_COMMIT}"
    )

    model = ResNet56()
    protocol_check(model)
    model = model.to(device)
    train_dataset, test_dataset = build_datasets(args)
    train_loader, test_loader = build_loaders(train_dataset, test_dataset, args)

    optimizer = torch.optim.SGD(
        parameter_groups(model),
        lr=BASE_LR,
        momentum=MOMENTUM,
        dampening=0.0,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PLANNED_EPOCHS, eta_min=0.0
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    scaler = create_grad_scaler(amp_enabled)

    config_payload = {
        "protocol": LOCKED_PROTOCOL,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "amp": amp_enabled,
            "num_workers": args.num_workers,
            "actual_epochs": epochs_to_run,
        },
    }
    atomic_json_save(config_payload, config_path)
    write_metrics_header(metrics_path)

    log(
        "[PROTOCOL] input=32 optimizer=SGD lr=0.1 momentum=0.9 "
        "nesterov=True weight_decay=0.0005 warmup=0 cosine_to=0"
    )
    log(
        "[AUG] official_strong=True color_jitter_arg=0.4 "
        "randaugment=rand-m9-mstd0.5-inc1 random_erasing=0.25 "
        "normalization=ImageNet"
    )
    log(
        "[AUG] realized=timm1.0.27_random_resized_crop+bicubic+flip+"
        "randaugment+normalize+random_erasing (no separate ColorJitter op)"
    )
    log(
        f"[MODEL] teacher_params={count_parameters(model):,} "
        f"architecture=CIFAR-ResNet56"
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
            with autocast_context(amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.size(0)
            loss_sum += float(loss.detach().item()) * batch_size
            correct += top1_correct(logits.detach(), targets)
            total += batch_size

        latest_top1 = evaluate(model, test_loader, device, amp_enabled)
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
            model,
            optimizer,
            epoch,
            latest_top1,
            best_top1,
            args,
            epoch_times,
        )
        atomic_torch_save(payload, latest_path)
        if is_best:
            atomic_torch_save(payload, best_path)
        if is_closest:
            atomic_torch_save(payload, closest_path)

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
            "estimated_300_seconds": estimated_full,
            "estimated_300_human": format_duration(estimated_full),
            "elapsed_seconds": elapsed_seconds,
            "paths": {
                "best": str(best_path.resolve()),
                "latest": str(latest_path.resolve()),
                "closest_to_reference": str(closest_path.resolve()),
            },
            "protocol": LOCKED_PROTOCOL,
        }
        atomic_json_save(partial_summary, summary_path)

        log(
            f"[TEACHER][{epoch:03d}/{epochs_to_run:03d}] "
            f"loss={train_loss:.4f} train_acc={train_top1:.2f}% "
            f"test_acc={latest_top1:.2f}% best={best_top1:.2f}% "
            f"closest_ref={closest_top1:.2f}% lr={current_lr:.8f} "
            f"time={epoch_seconds:.1f}s avg_epoch={average_epoch:.1f}s "
            f"est_300={format_duration(estimated_full)} "
            f"elapsed={format_duration(elapsed_seconds)}"
            + (" saved_best" if is_best else "")
            + (" saved_closest" if is_closest else "")
        )

    total_elapsed = time.time() - start_time
    average_epoch = sum(epoch_times) / len(epoch_times)
    estimated_full = average_epoch * PLANNED_EPOCHS
    hashes = {
        "best": sha256_file(best_path),
        "latest": sha256_file(latest_path),
        "closest_to_reference": sha256_file(closest_path),
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
        "estimated_300_human": format_duration(estimated_full),
        "elapsed_seconds": total_elapsed,
        "elapsed_human": format_duration(total_elapsed),
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
    atomic_json_save(final_summary, summary_path)

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
        f"estimated_300_teacher={format_duration(estimated_full)} "
        f"elapsed={format_duration(total_elapsed)}"
    )
    log(f"[FINAL_RESULT] best_checkpoint={best_path.resolve()}")
    log(f"[FINAL_RESULT] latest_checkpoint={latest_path.resolve()}")
    log(f"[FINAL_RESULT] closest_checkpoint={closest_path.resolve()}")
    log(f"[FINAL_RESULT] summary={summary_path.resolve()}")
    log(f"[SHA256] best={hashes['best']}")
    log(f"[SHA256] latest={hashes['latest']}")
    log(f"[SHA256] closest={hashes['closest_to_reference']}")
    log("[DONE] Teacher training completed successfully; resources may be released.")


def main() -> None:
    try:
        train(parse_args())
    except Exception as error:
        log("=" * 80)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] Teacher training did not complete.")
        raise


if __name__ == "__main__":
    main()
