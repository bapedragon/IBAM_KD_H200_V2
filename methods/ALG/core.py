#!/usr/bin/env python3
"""Train DeiT-Ti with researcher-synchronized Adaptive Locality Guidance."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.transforms import InterpolationMode


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.ALG.official_lg import (
    OFFICIAL_LG_COMMIT,
    STUDENT_BLOCK_INDICES,
    STUDENT_CHANNELS,
    TEACHER_CHANNELS,
    LocalityGuidance,
)
from methods.KD.core import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    STUDENT_MODELS,
    atomic_torch_save,
    autocast_context,
    count_parameters,
    create_grad_scaler,
    ensure_timm,
    evaluate,
    format_duration,
    log,
    public_args,
    seed_everything,
    top1_correct,
)
from teachers.train_teacher_cifar100 import official_test_transform
from teachers.verify_checkpoints import DEFAULT_CHECKPOINT_ROOT, load_teacher


ALG_PAPER_DOI = "10.1109/TNNLS.2024.3515076"
ALG_PAPER_REFERENCE_TOP1 = 83.50
ALG_PAPER_LG_TOP1 = 83.26
ALG_PAPER_BASELINE_TOP1 = 80.65
ALG_PAPER_STOP_EPOCH = 108
TEACHER_IMAGE_SIZE = 32
STUDENT_IMAGE_SIZE = 224


class AdaptiveGuidanceController:
    """Researcher-supplied ALG controller on epoch-average LG loss.

    This is the same three-case derivative implementation used by the
    researcher-synchronized Ours run.  It waits for ``warm_up`` epochs and
    permanently disables guidance when the smoothed derivative is strictly
    greater than ``threshold``.  The supplied implementation has no extra
    descent-first guard.
    """

    def __init__(
        self,
        *,
        beta: float,
        threshold: float,
        smoothing_window: int,
        warm_up: int,
    ) -> None:
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.smoothing_window = int(smoothing_window)
        self.warm_up = int(warm_up)
        self.active = True
        self.stop_epoch: int | None = None
        self.guidance_loss_history: list[float] = []
        self.derivative_history: list[float | None] = []
        self.smoothed_derivative_history: list[float | None] = []
        self.beta_history: list[float] = []

    def beta_for_epoch(self, epoch: int) -> float:
        expected = len(self.beta_history) + 1
        if epoch != expected:
            raise ValueError(f"Expected beta request for epoch {expected}, got {epoch}")
        value = self.beta if self.active else 0.0
        self.beta_history.append(value)
        return value

    def _delta_at(self, epoch: int) -> float | None:
        """Return the researcher code's one-based delta at ``epoch``."""

        if epoch < 2 or len(self.guidance_loss_history) < epoch:
            return None
        if epoch <= self.smoothing_window:
            previous_mean = (
                sum(self.guidance_loss_history[: epoch - 1]) / (epoch - 1)
            )
            return (
                self.guidance_loss_history[epoch - 1] - previous_mean
            ) / epoch
        return (
            self.guidance_loss_history[epoch - 1]
            - self.guidance_loss_history[epoch - self.smoothing_window - 1]
        ) / self.smoothing_window

    def _compute_smoothed_derivative(self, epoch: int) -> float | None:
        """Port the three cases shown in the researcher trainer verbatim."""

        if epoch < 2 or len(self.guidance_loss_history) < 2:
            return None

        if epoch <= self.smoothing_window:
            deltas = [self._delta_at(index) for index in range(2, epoch + 1)]
            values = [float(value) for value in deltas if value is not None]
            return sum(values) / len(values) if values else None

        if epoch < 2 * self.smoothing_window:
            first = epoch - self.smoothing_window + 1
            deltas = [
                self._delta_at(index) for index in range(first, epoch + 1)
            ]
            values = [float(value) for value in deltas if value is not None]
            return sum(values) / len(values) if values else None

        total = 0.0
        first = epoch - self.smoothing_window + 1
        for index in range(first, epoch + 1):
            total += (
                self.guidance_loss_history[index - 1]
                - self.guidance_loss_history[
                    index - self.smoothing_window - 1
                ]
            )
        return total / (self.smoothing_window**2)

    def observe(self, epoch: int, guidance_loss: float) -> dict[str, Any]:
        expected = len(self.guidance_loss_history) + 1
        if epoch != expected:
            raise ValueError(f"Expected ALG observation for epoch {expected}, got {epoch}")
        if not self.active:
            raise ValueError(
                "ALG observations stop permanently after guidance is disabled"
            )

        self.guidance_loss_history.append(float(guidance_loss))
        derivative = self._delta_at(epoch)
        self.derivative_history.append(derivative)
        smoothed_derivative = (
            None
            if epoch < self.warm_up
            else self._compute_smoothed_derivative(epoch)
        )
        self.smoothed_derivative_history.append(smoothed_derivative)

        if (
            self.active
            and smoothed_derivative is not None
            and smoothed_derivative > self.threshold
        ):
            self.active = False
            self.stop_epoch = epoch
        return self.state_dict()

    def state_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "implementation": "researcher_screenshot_2026-07-21",
            "observed_signal": "complete_alg_lg_loss",
            "epoch_numbering": "one_based_adapter",
            "threshold": self.threshold,
            "smoothing_window": self.smoothing_window,
            "warm_up": self.warm_up,
            "active": self.active,
            "stop_epoch": self.stop_epoch,
            "guidance_loss_history": list(self.guidance_loss_history),
            "derivative_history": list(self.derivative_history),
            "smoothed_derivative_history": list(
                self.smoothed_derivative_history
            ),
            "beta_history": list(self.beta_history),
        }


def install_signal_handlers() -> None:
    def handle_signal(signum: int, frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        log("=" * 72)
        log(f"[FATAL][SIGNAL] Received {signal_name}; external termination requested.")
        if frame is not None:
            traceback.print_stack(frame)
        log("[FATAL] ALG training was interrupted before normal completion.")
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_signal)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("chaoyang",), default="chaoyang")
    parser.add_argument("--student", choices=("deit_ti",), default="deit_ti")
    parser.add_argument("--protocol-name", type=str, default="manual")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timing-run", action="store_true")
    parser.add_argument("--student-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=STUDENT_IMAGE_SIZE)
    parser.add_argument("--teacher-image-size", type=int, default=TEACHER_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--smoke-train-samples", type=int, default=1024)
    parser.add_argument("--smoke-test-samples", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--warmup-factor", type=float, default=0.001)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=2.5)
    parser.add_argument("--alg-threshold", type=float, default=-0.02)
    parser.add_argument("--alg-smoothing-window", type=int, default=50)
    parser.add_argument(
        "--alg-warmup-epochs",
        type=int,
        default=20,
        help=(
            "Epoch before which the researcher controller cannot stop "
            "guidance (researcher implementation default: 20)."
        ),
    )
    parser.add_argument(
        "--base-protocol",
        choices=("lg_official", "draft_common"),
        default="lg_official",
        help=(
            "lg_official uses the audited public LG/ALG loader and scheduler; "
            "draft_common reproduces the historical Ours 81.11%% shared base."
        ),
    )
    parser.add_argument(
        "--eval-resize-mode",
        choices=("direct", "center_crop"),
        default="direct",
    )
    parser.add_argument(
        "--max-teacher-runtime-gap-pp",
        type=float,
        default=5.0,
    )
    parser.add_argument("--allow-teacher-runtime-gap", action="store_true")
    parser.add_argument(
        "--amp",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Official LG configuration uses FP32; enable only as a labeled deviation.",
    )
    return parser.parse_args()


def finalize_args(args: argparse.Namespace) -> None:
    args.planned_epochs = args.student_epochs
    if args.timing_run:
        args.student_epochs = 2
    if args.data_dir is None:
        args.data_dir = Path("/app/data/chaoyang")
    if args.run_name is None:
        suffix = (
            "timing_2ep"
            if args.timing_run
            else ("smoke" if args.smoke else f"{args.student_epochs}ep")
        )
        args.run_name = f"alg_chaoyang_deit_ti_{suffix}"

    positive = (
        "student_epochs",
        "batch_size",
        "eval_batch_size",
        "image_size",
        "teacher_image_size",
        "smoke_train_samples",
        "smoke_test_samples",
        "lr",
        "warmup_factor",
        "beta",
        "alg_smoothing_window",
    )
    for field in positive:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.min_lr < 0:
        raise ValueError("--min-lr must be non-negative")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be non-negative")
    if args.alg_warmup_epochs < 0:
        raise ValueError("--alg-warmup-epochs must be non-negative")
    if args.min_lr > args.lr:
        raise ValueError("--min-lr must not exceed --lr")
    if args.alg_threshold >= 0:
        raise ValueError("--alg-threshold must be negative")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if not 0.0 <= args.drop_path_rate < 1.0:
        raise ValueError("--drop-path-rate must be in [0, 1)")
    if args.image_size != STUDENT_IMAGE_SIZE:
        raise ValueError("ALG paper protocol requires --image-size 224")
    if args.teacher_image_size != TEACHER_IMAGE_SIZE:
        raise ValueError("LG/ALG ResNet56 guidance requires --teacher-image-size 32")


def build_alg_loaders(
    args: argparse.Namespace,
    device: torch.device,
    timm: Any,
) -> tuple[Any, Any]:
    # Exact public LG strong-augmentation arguments from
    # pycls/datasets/transforms.py at OFFICIAL_LG_COMMIT.
    train_transform = timm.data.create_transform(
        input_size=(3, STUDENT_IMAGE_SIZE, STUDENT_IMAGE_SIZE),
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        interpolation="bicubic",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (STUDENT_IMAGE_SIZE, STUDENT_IMAGE_SIZE),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    if args.dataset == "cifar100":
        from teachers.train_teacher_cifar100 import ensure_cifar100

        log(f"[DATA] CIFAR-100 root={args.data_dir.expanduser().resolve()}")
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
        split_description = "train:official_train eval:official_test"
    elif args.dataset == "chaoyang":
        from teachers.train_teacher_chaoyang import (
            ChaoyangDataset,
            resolve_dataset_root,
        )

        dataset_root = resolve_dataset_root(args.data_dir)
        log(f"[DATA] requested_root={args.data_dir.expanduser().resolve()}")
        log(f"[DATA] resolved_root={dataset_root}")
        log("[DATA] Validating official Chaoyang train/test splits and labels")
        train_dataset = ChaoyangDataset(
            dataset_root,
            "train",
            train_transform,
        )
        test_dataset = ChaoyangDataset(
            dataset_root,
            "test",
            test_transform,
        )
        split_description = "train:official_train eval:official_test"
    else:
        raise ValueError(
            "The audited public LG loader currently supports CIFAR-100 and "
            "Chaoyang only"
        )
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
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    log(f"[DATA] train_samples={len(train_dataset)} test_samples={len(test_dataset)}")
    log(
        f"[DATA] student_image=224 teacher_image=32 train_batch={args.batch_size} "
        f"eval_batch={args.eval_batch_size} num_workers={args.num_workers} "
        f"drop_last_train=True smoke={args.smoke} "
        f"split={split_description}"
    )
    return train_loader, test_loader


def create_draft_common_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
    base_lr: float,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    """Historical shared scheduler used by the 81.11% Ours run."""

    effective_warmup = warmup_epochs if epochs > warmup_epochs else 0
    minimum_ratio = min_lr / base_lr

    def lr_multiplier(epoch_index: int) -> float:
        if effective_warmup and epoch_index < effective_warmup:
            return (epoch_index + 1) / effective_warmup
        cosine_epochs = max(1, epochs - effective_warmup)
        progress = min(
            max((epoch_index - effective_warmup) / cosine_epochs, 0.0), 1.0
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    return scheduler, effective_warmup


def build_native_teacher_audit_loader(
    args: argparse.Namespace,
    device: torch.device,
) -> Any:
    from teachers.train_teacher_chaoyang import (
        ChaoyangDataset,
        resolve_dataset_root,
    )

    dataset_root = resolve_dataset_root(args.data_dir)
    dataset: Dataset[Any] = ChaoyangDataset(
        dataset_root,
        "test",
        official_test_transform(),
    )
    if args.smoke:
        generator = torch.Generator().manual_seed(args.seed + 1)
        indexes = torch.randperm(len(dataset), generator=generator)[
            : min(args.smoke_test_samples, len(dataset))
        ]
        dataset = Subset(dataset, indexes.tolist())
    return DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def student_view_to_teacher_view(images: torch.Tensor) -> torch.Tensor:
    """Official LG behavior: resize the already ImageNet-normalized view."""

    return F.interpolate(
        images,
        size=(TEACHER_IMAGE_SIZE, TEACHER_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )


def forward_teacher_features(
    teacher: torch.nn.Module,
    images: torch.Tensor,
) -> list[torch.Tensor]:
    return list(teacher.forward_features(student_view_to_teacher_view(images)))


@torch.inference_mode()
def evaluate_teacher_native(
    teacher: torch.nn.Module,
    loader: Any,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    teacher.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(amp_enabled):
            logits = teacher(images)
        correct += top1_correct(logits, targets)
        total += targets.size(0)
    return 100.0 * correct / max(1, total)


@torch.inference_mode()
def evaluate_teacher_shared_view(
    teacher: torch.nn.Module,
    loader: Any,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    teacher.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(amp_enabled):
            logits = teacher(student_view_to_teacher_view(images))
        correct += top1_correct(logits, targets)
        total += targets.size(0)
    return 100.0 * correct / max(1, total)


def create_student(
    timm: Any,
    num_classes: int,
    drop_path_rate: float,
) -> torch.nn.Module:
    return timm.create_model(
        STUDENT_MODELS["deit_ti"],
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )


def forward_student_features(
    student: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    final_tokens, features = student.forward_intermediates(
        images,
        indices=list(STUDENT_BLOCK_INDICES),
        norm=False,
        output_fmt="NCHW",
    )
    return list(features), student.forward_head(final_tokens)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    planned_epochs: int,
    lr: float,
    min_lr: float,
    warmup_epochs: int,
    warmup_factor: float,
) -> Any:
    from timm.scheduler import CosineLRScheduler

    return CosineLRScheduler(
        optimizer,
        t_initial=planned_epochs,
        lr_min=min_lr,
        warmup_t=warmup_epochs,
        warmup_lr_init=lr * warmup_factor,
    )


def train_one_epoch(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    guidance: LocalityGuidance,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
    beta: float,
) -> tuple[float, float, float, float]:
    student.train()
    guidance.train()
    teacher.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_lg = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        teacher_features: list[torch.Tensor] | None = None
        if beta > 0.0:
            with torch.no_grad(), autocast_context(amp_enabled):
                teacher_features = forward_teacher_features(teacher, images)
        with autocast_context(amp_enabled):
            student_features, logits = forward_student_features(student, images)
            ce = F.cross_entropy(
                logits,
                targets,
                label_smoothing=args.label_smoothing,
            )
            if teacher_features is None:
                lg_loss = ce.new_zeros(())
                loss = ce
            else:
                lg_loss, _, _ = guidance(student_features, teacher_features)
                loss = ce + beta * lg_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_ce += float(ce.detach()) * batch_size
        total_lg += float(lg_loss.detach()) * batch_size
        correct += top1_correct(logits.detach(), targets)

    denominator = max(1, total)
    return (
        total_loss / denominator,
        total_ce / denominator,
        total_lg / denominator,
        100.0 * correct / denominator,
    )


def checkpoint_payload(
    student: torch.nn.Module,
    guidance: LocalityGuidance,
    epoch: int,
    accuracy: float,
    best_accuracy: float,
    args: argparse.Namespace,
    teacher_spec: dict[str, Any],
    controller: AdaptiveGuidanceController,
) -> dict[str, Any]:
    return {
        "model": student.state_dict(),
        "model_state": student.state_dict(),
        "alg_guidance": guidance.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
        "best_accuracy": best_accuracy,
        "method": "ALG",
        "dataset": "chaoyang",
        "student": "deit_ti",
        "timm_model": STUDENT_MODELS["deit_ti"],
        "num_classes": NUM_CLASSES["chaoyang"],
        "teacher": teacher_spec,
        "controller": controller.state_dict(),
        "official_lg_commit": OFFICIAL_LG_COMMIT,
        "alg_paper_doi": ALG_PAPER_DOI,
        "args": public_args(args),
    }


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    teacher_spec: dict[str, Any],
    latest_epoch: int,
    best_accuracy: float,
    latest_accuracy: float,
    epoch_times: list[float],
    elapsed: float,
    controller: AdaptiveGuidanceController,
    teacher_native_top1: float,
    teacher_shared_top1: float,
) -> None:
    average_epoch = sum(epoch_times) / max(1, len(epoch_times))
    summary = {
        "status": "complete" if latest_epoch == args.student_epochs else "running",
        "method": "ALG",
        "dataset": "chaoyang",
        "student": "deit_ti",
        "teacher": teacher_spec,
        "objective": "CE + beta * LG while researcher ALG controller is active",
        "official_lg_commit": OFFICIAL_LG_COMMIT,
        "alg_paper_doi": ALG_PAPER_DOI,
        "controller": controller.state_dict(),
        "teacher_native_top1": teacher_native_top1,
        "teacher_shared_view_top1": teacher_shared_top1,
        "student_epochs": args.student_epochs,
        "latest_epoch": latest_epoch,
        "best_top1": best_accuracy,
        "latest_top1": latest_accuracy,
        "paper_alg_top1": ALG_PAPER_REFERENCE_TOP1,
        "gap_to_paper_alg_pp": best_accuracy - ALG_PAPER_REFERENCE_TOP1,
        "paper_lg_top1": ALG_PAPER_LG_TOP1,
        "paper_baseline_top1": ALG_PAPER_BASELINE_TOP1,
        "paper_guidance_stop_epoch": ALG_PAPER_STOP_EPOCH,
        "epoch_times": epoch_times,
        "avg_epoch_seconds": average_epoch,
        "planned_epochs": args.planned_epochs,
        "estimated_planned_seconds": average_epoch * args.planned_epochs,
        "estimated_planned_human": format_duration(
            average_epoch * args.planned_epochs
        ),
        "elapsed_seconds": elapsed,
        "elapsed_human": format_duration(elapsed),
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
    torch.backends.cudnn.benchmark = False
    timm = ensure_timm()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = run_dir / "student_best.pt"
    latest_checkpoint = run_dir / "student_latest.pt"
    summary_path = run_dir / "summary.json"

    log("=" * 72)
    log("ADAPTIVE LOCALITY GUIDANCE / RESNET56 -> DEIT-TI")
    log("=" * 72)
    log(
        f"[ENV] python={platform.python_version()} torch={torch.__version__} "
        f"torchvision={__import__('torchvision').__version__} timm={timm.__version__}"
    )
    log(
        f"[ENV] cuda_available={torch.cuda.is_available()} "
        f"cuda_device_count={torch.cuda.device_count()}"
    )
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        log(f"[ENV] gpu_name={torch.cuda.get_device_name(0)}")
        log(f"[ENV] gpu_memory_gib={properties.total_memory / (1024**3):.2f}")
    log(
        f"[ENV] device={device} amp={amp_enabled} seed={args.seed} "
        "cudnn_benchmark=False"
    )
    log(f"[PATH] data_dir={args.data_dir.resolve()}")
    log(f"[PATH] teacher_root={args.teacher_root.resolve()}")
    log(f"[PATH] run_dir={run_dir.resolve()}")
    log(
        f"[MODE] smoke={args.smoke} timing_run={args.timing_run} "
        f"student_epochs={args.student_epochs} planned_epochs={args.planned_epochs}"
    )
    log(
        f"[PROTOCOL] name={args.protocol_name} optimizer=AdamW lr={args.lr} "
        f"min_lr={args.min_lr} weight_decay={args.weight_decay} "
        f"warmup={args.warmup_epochs} warmup_factor={args.warmup_factor} "
        f"cosine batch={args.batch_size} eval_batch={args.eval_batch_size} "
        f"student_image=224 teacher_image=32 drop_path={args.drop_path_rate} "
        f"label_smoothing={args.label_smoothing} pretrained=False "
        f"base={args.base_protocol} eval_resize={args.eval_resize_mode}"
    )
    if args.base_protocol == "lg_official":
        log(
            "[AUGMENT] color_jitter=0.4 auto_augment=rand-m9-mstd0.5-inc1 "
            "random_erasing=0.25/pixel/1 interpolation=bicubic "
            "normalization=ImageNet drop_last_train=True"
        )
    else:
        log(
            "[AUGMENT] RandomResizedCrop(scale=0.8..1.0)+HorizontalFlip "
            "interpolation=bicubic normalization=ImageNet "
            "drop_last_train=False eval=Resize256+CenterCrop224"
        )
    log(
        f"[ALG] loss=CE+beta*LG_while_controller_active "
        f"beta={args.beta} tau={args.alg_threshold} "
        f"smoothing_window={args.alg_smoothing_window} "
        f"controller_warm_up={args.alg_warmup_epochs} "
        "stop_condition=smoothed_derivative>tau "
        "descent_guard=False observed_signal=complete_alg_lg_loss"
    )
    log(
        "[LG] teacher_stages=(0,1,2) student_blocks=(0,6,11) "
        "projection=1x1 grid=larger_of_teacher_student interpolation=bilinear "
        "stage_loss=elementwise_MSE sum_stages=True"
    )
    log(
        f"[REFERENCE] ALG_paper_Chaoyang_DeiT={ALG_PAPER_REFERENCE_TOP1:.2f}% "
        f"LG={ALG_PAPER_LG_TOP1:.2f}% baseline={ALG_PAPER_BASELINE_TOP1:.2f}% "
        f"paper_guidance_stop_epoch={ALG_PAPER_STOP_EPOCH}"
    )
    log(
        f"[SOURCE] ALG_DOI={ALG_PAPER_DOI} LG_repo_commit={OFFICIAL_LG_COMMIT}"
    )

    if args.base_protocol == "lg_official":
        train_loader, test_loader = build_alg_loaders(args, device, timm)
    else:
        from methods.KD.core import build_loaders

        train_loader, test_loader = build_loaders(args, device)
    native_loader = build_native_teacher_audit_loader(args, device)
    teacher, teacher_payload, teacher_spec = load_teacher(
        "chaoyang",
        device=device,
        checkpoint_root=args.teacher_root,
    )
    teacher_native_top1 = evaluate_teacher_native(
        teacher,
        native_loader,
        device,
        amp_enabled,
    )
    teacher_shared_top1 = evaluate_teacher_shared_view(
        teacher,
        test_loader,
        device,
        amp_enabled,
    )
    student = create_student(
        timm,
        NUM_CLASSES["chaoyang"],
        args.drop_path_rate,
    ).to(device)
    guidance = LocalityGuidance().to(device)

    with torch.no_grad():
        probe = torch.zeros(2, 3, STUDENT_IMAGE_SIZE, STUDENT_IMAGE_SIZE, device=device)
        student_probe, logits_probe = forward_student_features(student, probe)
        teacher_probe = forward_teacher_features(teacher, probe)
        lg_probe, aligned_student, aligned_teacher = guidance(
            student_probe,
            teacher_probe,
        )
    expected_student = [(2, STUDENT_CHANNELS, 14, 14)] * 3
    expected_teacher = [
        (2, TEACHER_CHANNELS[0], 32, 32),
        (2, TEACHER_CHANNELS[1], 16, 16),
        (2, TEACHER_CHANNELS[2], 8, 8),
    ]
    expected_aligned = [
        (2, TEACHER_CHANNELS[0], 32, 32),
        (2, TEACHER_CHANNELS[1], 16, 16),
        (2, TEACHER_CHANNELS[2], 14, 14),
    ]
    if [tuple(value.shape) for value in student_probe] != expected_student:
        raise RuntimeError(
            f"Unexpected student features: {[tuple(x.shape) for x in student_probe]}"
        )
    if [tuple(value.shape) for value in teacher_probe] != expected_teacher:
        raise RuntimeError(
            f"Unexpected teacher features: {[tuple(x.shape) for x in teacher_probe]}"
        )
    if [tuple(value.shape) for value in aligned_student] != expected_aligned:
        raise RuntimeError("Unexpected aligned student feature shapes")
    if [tuple(value.shape) for value in aligned_teacher] != expected_aligned:
        raise RuntimeError("Unexpected aligned teacher feature shapes")
    if tuple(logits_probe.shape) != (2, NUM_CLASSES["chaoyang"]):
        raise RuntimeError(f"Unexpected logits: {tuple(logits_probe.shape)}")
    if not bool(torch.isfinite(lg_probe)):
        raise RuntimeError("Non-finite ALG probe loss")

    checkpoint_top1 = float(teacher_payload["accuracy"])
    log(
        f"[TEACHER] selected={teacher_spec['selected_kind']} "
        f"epoch={teacher_payload['epoch']} top1={checkpoint_top1:.2f}% "
        f"sha256={teacher_spec['sha256']}"
    )
    log(
        f"[TEACHER_NATIVE_AUDIT] checkpoint_top1={checkpoint_top1:.2f}% "
        f"native_direct_32px_top1={teacher_native_top1:.2f}% "
        f"gap={teacher_native_top1 - checkpoint_top1:+.2f}pp"
    )
    log(
        f"[TEACHER_SHARED_VIEW] resize_224_to_32px_top1={teacher_shared_top1:.2f}% "
        f"gap_to_checkpoint={teacher_shared_top1 - checkpoint_top1:+.2f}pp "
        "diagnostic_only=True"
    )
    runtime_gap = teacher_native_top1 - checkpoint_top1
    if runtime_gap < -args.max_teacher_runtime_gap_pp:
        log(
            f"[TEACHER_NATIVE_AUDIT][WARN] gap exceeds "
            f"-{args.max_teacher_runtime_gap_pp:.1f}pp"
        )
        if not (args.smoke or args.timing_run or args.allow_teacher_runtime_gap):
            raise RuntimeError(
                "Teacher native accuracy audit failed before the full run."
            )
    log(
        f"[MODEL] teacher_params={count_parameters(teacher):,} "
        f"student={STUDENT_MODELS['deit_ti']} "
        f"student_params={count_parameters(student):,} "
        f"lg_trainable_params={count_parameters(guidance):,}"
    )
    log(
        f"[FEATURE_CHECK] teacher_raw={expected_teacher} "
        f"student={expected_student} aligned={expected_aligned} "
        f"probe_lg={float(lg_probe):.6f}"
    )

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(guidance.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.base_protocol == "lg_official":
        scheduler = create_scheduler(
            optimizer,
            planned_epochs=args.planned_epochs,
            lr=args.lr,
            min_lr=args.min_lr,
            warmup_epochs=args.warmup_epochs,
            warmup_factor=args.warmup_factor,
        )
        warmup_description = (
            f"official_warmup_start_lr={args.lr * args.warmup_factor:.8g}"
        )
    else:
        scheduler, effective_warmup = create_draft_common_scheduler(
            optimizer,
            args.planned_epochs,
            args.warmup_epochs,
            args.min_lr,
            args.lr,
        )
        warmup_description = (
            f"common_effective_warmup={effective_warmup} "
            f"first_epoch_lr={args.lr / max(1, effective_warmup):.8g}"
        )
    scaler = create_grad_scaler(amp_enabled)
    controller = AdaptiveGuidanceController(
        beta=args.beta,
        threshold=args.alg_threshold,
        smoothing_window=args.alg_smoothing_window,
        warm_up=args.alg_warmup_epochs,
    )
    log(
        f"[STUDENT] epochs={args.student_epochs} planned={args.planned_epochs} "
        f"initial_optimizer_lr={optimizer.param_groups[0]['lr']:.8g} "
        f"{warmup_description}"
    )

    best_accuracy = 0.0
    latest_accuracy = 0.0
    epoch_times: list[float] = []
    training_start = time.time()
    for epoch_index in range(args.student_epochs):
        epoch = epoch_index + 1
        epoch_start = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]
        beta = controller.beta_for_epoch(epoch)
        loss, ce, lg_loss, train_accuracy = train_one_epoch(
            student,
            teacher,
            guidance,
            train_loader,
            optimizer,
            scaler,
            device,
            args,
            amp_enabled,
            beta,
        )
        if beta > 0.0:
            state = controller.observe(epoch, lg_loss)
            raw_derivative = state["derivative_history"][-1]
            smoothed_derivative = state["smoothed_derivative_history"][-1]
        else:
            raw_derivative = None
            smoothed_derivative = None
        raw_text = "n/a" if raw_derivative is None else f"{raw_derivative:.6f}"
        smooth_text = (
            "n/a"
            if smoothed_derivative is None
            else f"{smoothed_derivative:.6f}"
        )
        if beta > 0.0 and not controller.active:
            log(
                f"[BETA_TRANSITION] guidance disabled after epoch={epoch} "
                f"LG={lg_loss:.6f} raw_derivative={raw_text} "
                f"smoothed_derivative={smooth_text} threshold={args.alg_threshold}; "
                "subsequent epochs are CE-only."
            )

        latest_accuracy = evaluate(student, test_loader, device, amp_enabled)
        epoch_seconds = time.time() - epoch_start
        epoch_times.append(epoch_seconds)
        previous_best = best_accuracy
        best_accuracy = max(best_accuracy, latest_accuracy)
        payload = checkpoint_payload(
            student,
            guidance,
            epoch,
            latest_accuracy,
            best_accuracy,
            args,
            teacher_spec,
            controller,
        )
        atomic_torch_save(payload, latest_checkpoint)
        saved_best = latest_accuracy >= previous_best
        if saved_best:
            atomic_torch_save(payload, best_checkpoint)

        elapsed = time.time() - training_start
        write_summary(
            summary_path,
            args=args,
            teacher_spec=teacher_spec,
            latest_epoch=epoch,
            best_accuracy=best_accuracy,
            latest_accuracy=latest_accuracy,
            epoch_times=epoch_times,
            elapsed=elapsed,
            controller=controller,
            teacher_native_top1=teacher_native_top1,
            teacher_shared_top1=teacher_shared_top1,
        )
        average_epoch = sum(epoch_times) / len(epoch_times)
        suffix = " saved_best" if saved_best else ""
        log(
            f"[ALG][{epoch:03d}/{args.student_epochs:03d}] loss={loss:.4f} "
            f"ce={ce:.4f} lg={lg_loss:.4f} beta={beta:.4f} "
            f"guidance_active_next={controller.active} "
            f"raw_derivative={raw_text} smoothed_derivative={smooth_text} "
            f"guidance_stop_epoch={controller.stop_epoch} "
            f"train_acc={train_accuracy:.2f}% val_acc={latest_accuracy:.2f}% "
            f"best={best_accuracy:.2f}% lr={epoch_lr:.8g} "
            f"time={epoch_seconds:.1f}s avg_epoch={average_epoch:.1f}s "
            f"est_planned={format_duration(average_epoch * args.planned_epochs)} "
            f"elapsed={format_duration(elapsed)}{suffix}"
        )
        scheduler.step(epoch)

    elapsed = time.time() - training_start
    average_epoch = sum(epoch_times) / len(epoch_times)
    log("=" * 72)
    log(
        f"[FINAL_RESULT] alg_best_top1={best_accuracy:.2f}% "
        f"paper_alg_top1={ALG_PAPER_REFERENCE_TOP1:.2f}% "
        f"gap_to_paper={best_accuracy - ALG_PAPER_REFERENCE_TOP1:+.2f}pp"
    )
    log(
        f"[FINAL_RESULT] paper_lg_top1={ALG_PAPER_LG_TOP1:.2f}% "
        f"gain_over_paper_lg={best_accuracy - ALG_PAPER_LG_TOP1:+.2f}pp"
    )
    log(
        f"[BETA_FINAL] observed_stop_epoch={controller.stop_epoch} "
        f"paper_stop_epoch={ALG_PAPER_STOP_EPOCH} "
        f"guided_epochs={sum(value > 0 for value in controller.beta_history)} "
        f"ce_only_epochs={sum(value == 0 for value in controller.beta_history)}"
    )
    log(
        f"[TIMING] avg_epoch={average_epoch:.1f}s planned_epochs={args.planned_epochs} "
        f"estimated_total={format_duration(average_epoch * args.planned_epochs)} "
        f"elapsed={format_duration(elapsed)}"
    )
    log(f"[FINAL_RESULT] best_checkpoint={best_checkpoint.resolve()}")
    log(f"[FINAL_RESULT] latest_checkpoint={latest_checkpoint.resolve()}")
    log(f"[FINAL_RESULT] summary={summary_path.resolve()}")
    log("[DONE] ALG training completed successfully; resources may be released.")


def cli_main() -> None:
    try:
        main()
    except Exception as error:
        log("=" * 72)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] ALG training did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
