#!/usr/bin/env python3
"""Train ViT students with standard logit knowledge distillation."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import random
import signal
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100, Flowers102
from torchvision.transforms import InterpolationMode


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from teachers.train_teacher_cifar100 import ensure_cifar100
from teachers.verify_checkpoints import DEFAULT_CHECKPOINT_ROOT, load_teacher


TIMM_VERSION = "1.0.27"
STUDENT_MODELS = {
    "deit_ti": "deit_tiny_patch16_224",
    "convit": "convit_tiny",
    "pit": "pit_ti_224",
    "pvtv2": "pvt_v2_b0",
}
PENDING_OFFICIAL_INTEGRATION = ("cvt", "t2t_7", "t2t_14")
TEACHER_IMAGE_SIZE = 32
STUDENT_IMAGE_SIZE = 224
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
STUDENT_NORMALIZATION = {
    "cifar100": (CIFAR100_MEAN, CIFAR100_STD),
    "flowers102": (IMAGENET_MEAN, IMAGENET_STD),
    "chaoyang": (IMAGENET_MEAN, IMAGENET_STD),
}
NUM_CLASSES = {"cifar100": 100, "flowers102": 102, "chaoyang": 4}
VANILLA_TOP1 = {
    "cifar100": {
        "deit_ti": 65.08,
        "convit": 74.87,
        "cvt": 74.29,
        "pit": 73.16,
        "pvtv2": 77.21,
        "t2t_7": 68.00,
        "t2t_14": 69.93,
    },
    "flowers102": {
        "deit_ti": 50.06,
        "convit": 57.45,
        "cvt": 60.82,
        "pit": 56.12,
        "pvtv2": 67.89,
        "t2t_7": 66.14,
        "t2t_14": 64.95,
    },
    "chaoyang": {
        "deit_ti": 82.00,
        "convit": 80.93,
        "cvt": 80.04,
        "pit": 81.53,
        "pvtv2": 82.52,
        "t2t_7": 78.78,
        "t2t_14": 75.04,
    },
}


def log(message: str = "") -> None:
    print(message, flush=True)


def ensure_timm() -> Any:
    try:
        timm = importlib.import_module("timm")
    except ModuleNotFoundError:
        log(f"[BOOT] timm not found; installing timm=={TIMM_VERSION}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", f"timm=={TIMM_VERSION}"]
        )
        timm = importlib.import_module("timm")
        log("[BOOT] timm installation completed")
    if timm.__version__ != TIMM_VERSION:
        log(
            f"[BOOT][WARN] expected timm={TIMM_VERSION}, found timm={timm.__version__}; "
            "the model API may differ"
        )
    return timm


def install_signal_handlers() -> None:
    def handle_signal(signum: int, frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        log("=" * 72)
        log(f"[FATAL][SIGNAL] Received {signal_name}; external termination requested.")
        if frame is not None:
            traceback.print_stack(frame)
        log("[FATAL] KD training was interrupted before normal completion.")
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_signal)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(NUM_CLASSES), default="cifar100")
    parser.add_argument("--student", choices=tuple(STUDENT_MODELS), default="deit_ti")
    parser.add_argument(
        "--protocol-name",
        type=str,
        default="manual",
        help="Recorded name of the dataset-specific training protocol.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--timing-run",
        action="store_true",
        help="Use the full dataset for two epochs and estimate the planned full run.",
    )
    parser.add_argument("--student-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-train-samples", type=int, default=1024)
    parser.add_argument("--smoke-test-samples", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument(
        "--kd-weight",
        type=float,
        default=0.9,
        help="Weight alpha in (1-alpha)*CE + alpha*T^2*KL.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--amp",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use CUDA autocast when CUDA is available.",
    )
    return parser.parse_args()


def finalize_args(args: argparse.Namespace) -> None:
    args.planned_epochs = args.student_epochs
    if args.timing_run:
        args.student_epochs = 2
    if args.data_dir is None:
        args.data_dir = (
            Path("/app/data/chaoyang") if args.dataset == "chaoyang" else Path("./data")
        )
    if args.run_name is None:
        suffix = (
            "timing_2ep"
            if args.timing_run
            else ("smoke" if args.smoke else f"{args.student_epochs}ep")
        )
        args.run_name = f"kd_{args.dataset}_{args.student}_{suffix}"

    positive_fields = (
        "student_epochs",
        "batch_size",
        "image_size",
        "smoke_train_samples",
        "smoke_test_samples",
        "lr",
        "temperature",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be non-negative")
    if not 0.0 <= args.kd_weight <= 1.0:
        raise ValueError("--kd-weight must be in [0, 1]")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.image_size != 224:
        raise ValueError("The shared paper protocol requires --image-size 224")


def build_eval_transform(
    dataset: str,
    resize_mode: str = "center_crop",
) -> transforms.Compose:
    """Build the recorded student evaluation transform.

    Generic KD keeps the previously locked Resize(256)+CenterCrop(224)
    protocol. Ours can explicitly request the direct 224x224 resize used by
    the supplied/public locality-guidance code, which also preserves the full
    field of view before the shared image is downsampled for the 32px teacher.
    """

    if resize_mode not in {"center_crop", "direct"}:
        raise ValueError(
            "eval resize mode must be one of {'center_crop', 'direct'}, "
            f"got {resize_mode!r}"
        )
    student_mean, student_std = STUDENT_NORMALIZATION[dataset]
    if resize_mode == "direct":
        spatial_transforms: list[Any] = [
            transforms.Resize(
                (STUDENT_IMAGE_SIZE, STUDENT_IMAGE_SIZE),
                interpolation=InterpolationMode.BICUBIC,
            )
        ]
    else:
        resize_size = int(round(STUDENT_IMAGE_SIZE / 0.875))
        spatial_transforms = [
            transforms.Resize(
                resize_size,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(STUDENT_IMAGE_SIZE),
        ]
    return transforms.Compose(
        [
            *spatial_transforms,
            transforms.ToTensor(),
            transforms.Normalize(student_mean, student_std),
        ]
    )


def build_loaders(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    student_mean, student_std = STUDENT_NORMALIZATION[args.dataset]
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                STUDENT_IMAGE_SIZE,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(student_mean, student_std),
        ]
    )
    eval_resize_mode = getattr(args, "eval_resize_mode", "center_crop")
    test_transform = build_eval_transform(
        args.dataset,
        resize_mode=eval_resize_mode,
    )

    if args.dataset == "cifar100":
        log(f"[DATA] CIFAR-100 root={args.data_dir.resolve()}")
        log("[DATA] Preparing CIFAR-100 with verified mirror fallback")
        ensure_cifar100(args.data_dir)
        train_dataset: Dataset[Any] = CIFAR100(
            root=args.data_dir,
            train=True,
            transform=train_transform,
            download=False,
        )
        test_dataset: Dataset[Any] = CIFAR100(
            root=args.data_dir,
            train=False,
            transform=test_transform,
            download=False,
        )
        split_description = "train=official_train eval=official_test"
    elif args.dataset == "flowers102":
        from teachers.train_teacher_flowers import ensure_flowers, ensure_scipy

        log(f"[DATA] Oxford Flowers root={args.data_dir.resolve()}")
        log("[DATA] Preparing Oxford Flowers from official files with MD5 checks")
        ensure_scipy()
        ensure_flowers(args.data_dir)
        train_parts = [
            Flowers102(
                root=args.data_dir,
                split=split,
                transform=train_transform,
                download=False,
            )
            for split in ("train", "val")
        ]
        train_dataset = ConcatDataset(train_parts)
        test_dataset = Flowers102(
            root=args.data_dir,
            split="test",
            transform=test_transform,
            download=False,
        )
        split_description = "train=official_train+val eval=official_test"
    else:
        from teachers.train_teacher_chaoyang import (
            ChaoyangDataset,
            resolve_dataset_root,
        )

        dataset_root = resolve_dataset_root(args.data_dir)
        log(f"[DATA] requested_root={args.data_dir.expanduser().resolve()}")
        log(f"[DATA] resolved_root={dataset_root}")
        log("[DATA] Validating official Chaoyang train/test splits and labels")
        train_dataset = ChaoyangDataset(dataset_root, "train", train_transform)
        test_dataset = ChaoyangDataset(dataset_root, "test", test_transform)
        split_description = "train=official_train eval=official_test"
    if args.smoke:
        generator = torch.Generator().manual_seed(args.seed)
        train_indexes = torch.randperm(len(train_dataset), generator=generator)[
            : min(args.smoke_train_samples, len(train_dataset))
        ]
        test_generator = torch.Generator().manual_seed(args.seed + 1)
        test_indexes = torch.randperm(len(test_dataset), generator=test_generator)[
            : min(args.smoke_test_samples, len(test_dataset))
        ]
        train_dataset = Subset(train_dataset, train_indexes.tolist())
        test_dataset = Subset(test_dataset, test_indexes.tolist())

    def seed_worker(worker_id: int) -> None:
        del worker_id
        worker_seed = torch.initial_seed() % (2**32)
        random.seed(worker_seed)

    loader_generator = torch.Generator().manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )
    log(f"[DATA] train_samples={len(train_dataset)} test_samples={len(test_dataset)}")
    log(
        f"[DATA] student_image={STUDENT_IMAGE_SIZE} teacher_image={TEACHER_IMAGE_SIZE} "
        f"batch_size={args.batch_size} num_workers={args.num_workers} smoke={args.smoke} "
        f"eval_resize={eval_resize_mode} "
        f"{split_description}"
    )
    return train_loader, test_loader


def student_view_to_teacher_view(
    images: torch.Tensor,
    dataset: str = "cifar100",
) -> torch.Tensor:
    """Preserve crop/flip geometry while adapting size and normalization."""

    normalization_mean, normalization_std = STUDENT_NORMALIZATION[dataset]
    student_mean = images.new_tensor(normalization_mean).view(1, 3, 1, 1)
    student_std = images.new_tensor(normalization_std).view(1, 3, 1, 1)
    teacher_mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    teacher_std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    pixels = images * student_std + student_mean
    pixels = F.interpolate(
        pixels,
        size=(TEACHER_IMAGE_SIZE, TEACHER_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    return (pixels - teacher_mean) / teacher_std


def create_student(timm: Any, student_key: str, num_classes: int) -> nn.Module:
    timm_name = STUDENT_MODELS[student_key]
    try:
        return timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
    except Exception as error:
        available = timm.list_models(f"*{student_key.split('_')[0]}*")[:20]
        raise RuntimeError(
            f"Failed to create timm model {timm_name!r} for {student_key}; "
            f"nearby models={available}"
        ) from error


def create_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def top1_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == targets).sum().item())


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature**2)


def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
) -> tuple[float, float, float, float]:
    student.train()
    teacher.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad(), autocast_context(amp_enabled):
            teacher_logits = teacher(
                student_view_to_teacher_view(images, args.dataset)
            )
        with autocast_context(amp_enabled):
            student_logits = student(images)
            ce = F.cross_entropy(
                student_logits,
                targets,
                label_smoothing=args.label_smoothing,
            )
            distillation = kd_loss(
                student_logits.float(), teacher_logits.float(), args.temperature
            )
            loss = (1.0 - args.kd_weight) * ce + args.kd_weight * distillation

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_ce += float(ce.detach()) * batch_size
        total_kd += float(distillation.detach()) * batch_size
        correct += top1_correct(student_logits.detach(), targets)

    denominator = max(1, total)
    return (
        total_loss / denominator,
        total_ce / denominator,
        total_kd / denominator,
        100.0 * correct / denominator,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    amp_enabled: bool,
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
    return 100.0 * correct / max(1, total)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    effective_warmup = warmup_epochs if epochs > warmup_epochs else 0

    def lr_multiplier(epoch_index: int) -> float:
        if effective_warmup and epoch_index < effective_warmup:
            return (epoch_index + 1) / effective_warmup
        cosine_epochs = max(1, epochs - effective_warmup)
        progress = (epoch_index - effective_warmup) / cosine_epochs
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier), effective_warmup


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def public_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_payload(
    student: nn.Module,
    epoch: int,
    accuracy: float,
    best_accuracy: float,
    args: argparse.Namespace,
    teacher_spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": student.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
        "best_accuracy": best_accuracy,
        "method": "KD",
        "student": args.student,
        "timm_model": STUDENT_MODELS[args.student],
        "dataset": args.dataset,
        "num_classes": NUM_CLASSES[args.dataset],
        "teacher": teacher_spec,
        "args": public_args(args),
    }


def write_summary(
    path: Path,
    args: argparse.Namespace,
    teacher_spec: dict[str, Any],
    *,
    latest_epoch: int,
    best_accuracy: float,
    latest_accuracy: float,
    epoch_times: list[float],
    elapsed_seconds: float,
) -> None:
    average_epoch = sum(epoch_times) / max(1, len(epoch_times))
    estimated_300 = average_epoch * 300
    estimated_planned = average_epoch * args.planned_epochs
    summary = {
        "status": "complete" if latest_epoch == args.student_epochs else "running",
        "method": "KD",
        "dataset": args.dataset,
        "student": args.student,
        "timm_model": STUDENT_MODELS[args.student],
        "teacher": teacher_spec,
        "student_epochs": args.student_epochs,
        "latest_epoch": latest_epoch,
        "best_top1": best_accuracy,
        "latest_top1": latest_accuracy,
        "vanilla_top1": VANILLA_TOP1[args.dataset][args.student],
        "gain_over_vanilla_pp": best_accuracy - VANILLA_TOP1[args.dataset][args.student],
        "epoch_times": epoch_times,
        "avg_epoch_seconds": average_epoch,
        "planned_epochs": args.planned_epochs,
        "estimated_planned_seconds": estimated_planned,
        "estimated_planned_human": format_duration(estimated_planned),
        "estimated_300_seconds": estimated_300,
        "estimated_300_human": format_duration(estimated_300),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_human": format_duration(elapsed_seconds),
        "args": public_args(args),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    install_signal_handlers()
    args = parse_args()
    finalize_args(args)
    seed_everything(args.seed)
    timm = ensure_timm()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = run_dir / "student_best.pt"
    latest_checkpoint = run_dir / "student_latest.pt"
    summary_path = run_dir / "summary.json"

    log("=" * 72)
    log("STANDARD LOGIT KD / RESNET56 -> VIT STUDENT")
    log("=" * 72)
    log(
        f"[ENV] python={platform.python_version()} torch={torch.__version__} "
        f"timm={timm.__version__}"
    )
    log(
        f"[ENV] cuda_available={torch.cuda.is_available()} "
        f"cuda_device_count={torch.cuda.device_count()}"
    )
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        log(f"[ENV] gpu_name={torch.cuda.get_device_name(0)}")
        log(f"[ENV] gpu_memory_gib={properties.total_memory / (1024**3):.2f}")
    log(f"[ENV] device={device} amp={amp_enabled} seed={args.seed}")
    log(f"[PATH] data_dir={args.data_dir.resolve()}")
    log(f"[PATH] teacher_root={args.teacher_root.resolve()}")
    log(f"[PATH] run_dir={run_dir.resolve()}")
    log(
        f"[MODE] smoke={args.smoke} timing_run={args.timing_run} "
        f"student_epochs={args.student_epochs} planned_epochs={args.planned_epochs}"
    )
    log(
        f"[PROTOCOL] name={args.protocol_name} optimizer=AdamW "
        f"lr={args.lr} weight_decay={args.weight_decay} "
        f"warmup={args.warmup_epochs} cosine batch={args.batch_size} "
        f"image={args.image_size}"
    )
    log(
        f"[INPUT] shared_geometry=True student=224x224({args.dataset} norm) "
        "teacher=bilinear_downsample_to_32x32(ImageNet norm)"
    )
    log(
        f"[KD] loss=(1-{args.kd_weight})*CE+{args.kd_weight}*T^2*KL "
        f"temperature={args.temperature} label_smoothing={args.label_smoothing}"
    )
    log("[NOTE] KD temperature and loss weight are implementation choices, not specified in V2.")
    log(
        "[NOTE] CvT, T2T-7, and T2T-14 require their official model implementations; "
        "they are not exposed by timm==1.0.27."
    )

    train_loader, test_loader = build_loaders(args, device)
    teacher, teacher_payload, teacher_spec = load_teacher(
        args.dataset,
        device=device,
        checkpoint_root=args.teacher_root,
    )
    student = create_student(timm, args.student, NUM_CLASSES[args.dataset]).to(device)
    log(
        f"[TEACHER] selected={teacher_spec['selected_kind']} epoch={teacher_payload['epoch']} "
        f"top1={float(teacher_payload['accuracy']):.2f}% sha256={teacher_spec['sha256']}"
    )
    log(
        f"[MODEL] teacher_params={count_parameters(teacher):,} "
        f"student={STUDENT_MODELS[args.student]} student_params={count_parameters(student):,}"
    )

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler_horizon = args.planned_epochs if args.timing_run else args.student_epochs
    scheduler, effective_warmup = create_scheduler(
        optimizer,
        scheduler_horizon,
        args.warmup_epochs,
    )
    scaler = create_grad_scaler(amp_enabled)
    log(
        f"[STUDENT] optimizer=adamw lr={args.lr} weight_decay={args.weight_decay} "
        f"epochs={args.student_epochs} effective_warmup={effective_warmup}"
    )

    best_accuracy = 0.0
    latest_accuracy = 0.0
    epoch_times: list[float] = []
    training_start = time.time()

    for epoch_index in range(args.student_epochs):
        epoch = epoch_index + 1
        epoch_start = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]
        loss, ce, distillation, train_accuracy = train_one_epoch(
            student,
            teacher,
            train_loader,
            optimizer,
            scaler,
            device,
            args,
            amp_enabled,
        )
        latest_accuracy = evaluate(student, test_loader, device, amp_enabled)
        epoch_seconds = time.time() - epoch_start
        epoch_times.append(epoch_seconds)
        previous_best = best_accuracy
        best_accuracy = max(best_accuracy, latest_accuracy)

        payload = checkpoint_payload(
            student,
            epoch,
            latest_accuracy,
            best_accuracy,
            args,
            teacher_spec,
        )
        atomic_torch_save(payload, latest_checkpoint)
        saved_best = latest_accuracy >= previous_best
        if saved_best:
            atomic_torch_save(payload, best_checkpoint)

        elapsed = time.time() - training_start
        write_summary(
            summary_path,
            args,
            teacher_spec,
            latest_epoch=epoch,
            best_accuracy=best_accuracy,
            latest_accuracy=latest_accuracy,
            epoch_times=epoch_times,
            elapsed_seconds=elapsed,
        )
        average_epoch = sum(epoch_times) / len(epoch_times)
        suffix = " saved_best" if saved_best else ""
        log(
            f"[KD][{epoch:03d}/{args.student_epochs:03d}] loss={loss:.4f} "
            f"ce={ce:.4f} kd={distillation:.4f} train_acc={train_accuracy:.2f}% "
            f"val_acc={latest_accuracy:.2f}% best={best_accuracy:.2f}% "
            f"lr={epoch_lr:.6g} time={epoch_seconds:.1f}s "
            f"avg_epoch={average_epoch:.1f}s "
            f"est_planned={format_duration(average_epoch * args.planned_epochs)} "
            f"elapsed={format_duration(elapsed)}{suffix}"
        )
        scheduler.step()

    elapsed = time.time() - training_start
    average_epoch = sum(epoch_times) / len(epoch_times)
    vanilla = VANILLA_TOP1[args.dataset][args.student]
    log("=" * 72)
    log(
        f"[FINAL_RESULT] kd_best_top1={best_accuracy:.2f}% "
        f"vanilla_top1={vanilla:.2f}% gain_over_vanilla={best_accuracy - vanilla:+.2f}pp"
    )
    log(
        f"[TIMING] avg_epoch={average_epoch:.1f}s "
        f"planned_epochs={args.planned_epochs} "
        f"estimated_total={format_duration(average_epoch * args.planned_epochs)} "
        f"elapsed={format_duration(elapsed)}"
    )
    log(f"[FINAL_RESULT] best_checkpoint={best_checkpoint.resolve()}")
    log(f"[FINAL_RESULT] latest_checkpoint={latest_checkpoint.resolve()}")
    log(f"[FINAL_RESULT] summary={summary_path.resolve()}")
    log("[DONE] KD training completed successfully; resources may be released.")


def cli_main() -> None:
    try:
        main()
    except Exception as error:
        log("=" * 72)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] KD training did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
