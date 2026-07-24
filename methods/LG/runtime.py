#!/usr/bin/env python3
"""Official-LG training runtime shared by static LG and paper ALG."""

from __future__ import annotations

import argparse
import json
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100, Flowers102
from torchvision.transforms import InterpolationMode


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.LG.official_lg import (
    OFFICIAL_LG_COMMIT,
    STUDENT_BLOCK_INDICES,
    STUDENT_CHANNELS,
    TEACHER_CHANNELS,
    LocalityGuidance,
)
from methods.KD.core import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES as SHARED_NUM_CLASSES,
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
NUM_CLASSES = {**SHARED_NUM_CLASSES, "cub200": 200}
REFERENCE_TOP1 = {
    "cifar100": {
        "alg": 81.98,
        "alg_lg": 77.38,
        "alg_baseline": 65.08,
        "lg": 78.15,
        "baseline": 65.08,
    },
    "flowers102": {
        "alg": 68.54,
        "alg_lg": 67.02,
        "alg_baseline": 50.06,
        "lg": 68.50,
        "baseline": 50.06,
    },
    "chaoyang": {
        "alg": 83.50,
        "alg_lg": 83.26,
        "alg_baseline": 80.65,
        "lg": 84.20,
        "baseline": 82.00,
    },
    "cub200": {
        "alg": None,
        "alg_lg": None,
        "alg_baseline": None,
        "lg": None,
        "baseline": None,
    },
}
PAPER_GUIDANCE_STOP_EPOCH = {
    "cifar100": None,
    "flowers102": None,
    "chaoyang": 108,
    "cub200": None,
}
TEACHER_IMAGE_SIZE = 32
STUDENT_IMAGE_SIZE = 224


class AdaptiveGuidanceController:
    """ALG paper controller on the epoch-average official LG loss."""

    def __init__(
        self,
        *,
        beta: float,
        threshold: float,
        smoothing_window: int,
        warm_up: int,
        stop_comparison: str = "paper_ge",
        derivative_mode: str = "paper_equations",
    ) -> None:
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.smoothing_window = int(smoothing_window)
        self.warm_up = int(warm_up)
        if stop_comparison != "paper_ge":
            raise ValueError("Canonical ALG requires stop_comparison='paper_ge'")
        self.stop_comparison = stop_comparison
        if derivative_mode != "paper_equations":
            raise ValueError(
                "Canonical ALG requires derivative_mode='paper_equations'"
            )
        self.derivative_mode = derivative_mode
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
        """Return the ALG paper's one-based loss derivative at ``epoch``."""

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
        """Implement the three smoothing cases in ALG Eqs. (16)-(18)."""

        if epoch < 2 or len(self.guidance_loss_history) < 2:
            return None

        if epoch <= self.smoothing_window:
            deltas = [self._delta_at(index) for index in range(2, epoch + 1)]
            values = [float(value) for value in deltas if value is not None]
            if not values:
                return None
            # ALG Eq. (16) averages e derivative terms.  The i=1 term is
            # undefined algebraically because there is no previous epoch;
            # treating it as zero is the finite boundary completion that
            # preserves the equation's explicit 1/e normalization.
            return sum(values) / epoch

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

        threshold_crossed = (
            smoothed_derivative is not None
            and smoothed_derivative >= self.threshold
        )
        if self.active and threshold_crossed:
            self.active = False
            self.stop_epoch = epoch
        return self.state_dict()

    def state_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "implementation": "alg_paper_equations_10_to_19",
            "observed_signal": "complete_alg_lg_loss",
            "epoch_numbering": "one_based_adapter",
            "threshold": self.threshold,
            "smoothing_window": self.smoothing_window,
            "warm_up": self.warm_up,
            "stop_comparison": self.stop_comparison,
            "derivative_mode": self.derivative_mode,
            "active": self.active,
            "stop_epoch": self.stop_epoch,
            "guidance_loss_history": list(self.guidance_loss_history),
            "derivative_history": list(self.derivative_history),
            "smoothed_derivative_history": list(
                self.smoothed_derivative_history
            ),
            "beta_history": list(self.beta_history),
        }


class StaticGuidanceController:
    """Keep official LG active for the complete training schedule."""

    def __init__(self, *, beta: float) -> None:
        self.beta = float(beta)
        self.active = True
        self.stop_epoch: int | None = None
        self.guidance_loss_history: list[float] = []
        self.derivative_history: list[None] = []
        self.smoothed_derivative_history: list[None] = []
        self.beta_history: list[float] = []

    def beta_for_epoch(self, epoch: int) -> float:
        expected = len(self.beta_history) + 1
        if epoch != expected:
            raise ValueError(f"Expected beta request for epoch {expected}, got {epoch}")
        self.beta_history.append(self.beta)
        return self.beta

    def observe(self, epoch: int, guidance_loss: float) -> dict[str, Any]:
        expected = len(self.guidance_loss_history) + 1
        if epoch != expected:
            raise ValueError(f"Expected LG observation for epoch {expected}, got {epoch}")
        self.guidance_loss_history.append(float(guidance_loss))
        self.derivative_history.append(None)
        self.smoothed_derivative_history.append(None)
        return self.state_dict()

    def state_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "implementation": "official_lg_static_all_epochs",
            "active": True,
            "stop_epoch": None,
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
        log("[FATAL] LG/ALG training was interrupted before normal completion.")
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_signal)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=tuple(NUM_CLASSES),
        default="chaoyang",
    )
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
        default=0,
        help=(
            "Optional controller-only delay. Canonical ALG uses 0; the "
            "20-epoch warm-up belongs to the optimizer schedule."
        ),
    )
    parser.add_argument(
        "--alg-stop-comparison",
        choices=("paper_ge",),
        default="paper_ge",
        help=(
            "paper_ge implements ALG Eq. (19), where guidance is active only "
            "while the smoothed derivative is below tau."
        ),
    )
    parser.add_argument(
        "--alg-derivative-mode",
        choices=("paper_equations",),
        default="paper_equations",
        help=(
            "paper_equations uses ALG Eqs. (16)-(18), including the stated "
            "1/e early-epoch normalization."
        ),
    )
    parser.add_argument(
        "--base-protocol",
        choices=("lg_official",),
        default="lg_official",
        help=(
            "Use the audited public LG loader, model initialization, optimizer "
            "groups, and scheduler."
        ),
    )
    parser.add_argument(
        "--eval-resize-mode",
        choices=("direct",),
        default="direct",
    )
    parser.add_argument(
        "--flowers-split-policy",
        choices=("trainval_test_best", "official_three_way"),
        default="trainval_test_best",
        help=(
            "Flowers-102 split policy. official_three_way trains on the "
            "official 1,020-image train split, selects the checkpoint on the "
            "1,020-image val split, and evaluates the selected checkpoint once "
            "on the 6,149-image test split."
        ),
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
        args.data_dir = (
            Path("/app/data/chaoyang")
            if args.dataset == "chaoyang"
            else Path("./data")
        )
    if args.run_name is None:
        suffix = (
            "timing_2ep"
            if args.timing_run
            else ("smoke" if args.smoke else f"{args.student_epochs}ep")
        )
        args.run_name = f"{args.method.lower()}_{args.dataset}_deit_ti_{suffix}"

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
        raise ValueError("Official LG/ALG protocol requires --image-size 224")
    if args.teacher_image_size != TEACHER_IMAGE_SIZE:
        raise ValueError("LG/ALG ResNet56 guidance requires --teacher-image-size 32")
    if args.method == "ALG":
        canonical_controller = (
            args.alg_warmup_epochs == 0
            and args.alg_stop_comparison == "paper_ge"
            and args.alg_derivative_mode == "paper_equations"
        )
        if not canonical_controller:
            raise ValueError(
                "Canonical ALG requires controller warm-up 0, paper_ge, and "
                "paper_equations. Historical researcher-sync variants are not "
                "valid ALG entry points."
            )
    if args.base_protocol != "lg_official":
        raise ValueError(
            "LG and canonical ALG must use the official LG base protocol"
        )
    if (
        args.flowers_split_policy == "official_three_way"
        and args.dataset != "flowers102"
    ):
        raise ValueError(
            "--flowers-split-policy official_three_way is valid only for Flowers-102"
        )


def build_alg_loaders_with_final_test(
    args: argparse.Namespace,
    device: torch.device,
    timm: Any,
) -> tuple[Any, Any, Any | None]:
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
    elif args.dataset == "flowers102":
        from teachers.train_teacher_flowers import ensure_flowers, ensure_scipy

        log(f"[DATA] Oxford Flowers root={args.data_dir.expanduser().resolve()}")
        log("[DATA] Preparing Oxford Flowers from official files with MD5 checks")
        ensure_scipy()
        ensure_flowers(args.data_dir)
        if args.flowers_split_policy == "official_three_way":
            train_dataset = Flowers102(
                root=args.data_dir,
                split="train",
                transform=train_transform,
                download=False,
            )
            test_dataset = Flowers102(
                root=args.data_dir,
                split="val",
                transform=test_transform,
                download=False,
            )
            final_test_dataset: Dataset[Any] | None = Flowers102(
                root=args.data_dir,
                split="test",
                transform=test_transform,
                download=False,
            )
            split_description = (
                "train:official_train selection:official_val "
                "final:official_test_once"
            )
        else:
            train_dataset = ConcatDataset(
                [
                    Flowers102(
                        root=args.data_dir,
                        split=split,
                        transform=train_transform,
                        download=False,
                    )
                    for split in ("train", "val")
                ]
            )
            test_dataset = Flowers102(
                root=args.data_dir,
                split="test",
                transform=test_transform,
                download=False,
            )
            final_test_dataset = None
            split_description = "train:official_train+val eval:official_test"
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
    elif args.dataset == "cub200":
        from teachers.cub200_dataset import CUB200Dataset, ensure_cub200

        dataset_root = ensure_cub200(args.data_dir)
        log(f"[DATA] CUB-200-2011 root={dataset_root}")
        train_dataset = CUB200Dataset(
            dataset_root,
            split="train",
            transform=train_transform,
        )
        test_dataset = CUB200Dataset(
            dataset_root,
            split="test",
            transform=test_transform,
        )
        split_description = "train:official_train eval:official_test"
    else:
        raise ValueError(
            "The audited public LG loader supports CIFAR-100, Flowers-102, "
            "Chaoyang, and CUB-200 only"
        )
    if args.dataset != "flowers102":
        final_test_dataset = None
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
        if final_test_dataset is not None:
            final_test_generator = torch.Generator().manual_seed(args.seed + 2)
            final_test_indexes = torch.randperm(
                len(final_test_dataset), generator=final_test_generator
            )[: min(args.smoke_test_samples, len(final_test_dataset))]
            final_test_dataset = Subset(
                final_test_dataset, final_test_indexes.tolist()
            )

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
    final_test_loader = (
        DataLoader(
            final_test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        )
        if final_test_dataset is not None
        else None
    )
    if final_test_dataset is None:
        log(
            f"[DATA] train_samples={len(train_dataset)} "
            f"test_samples={len(test_dataset)}"
        )
    else:
        log(
            f"[DATA] train_samples={len(train_dataset)} "
            f"val_samples={len(test_dataset)} "
            f"test_samples={len(final_test_dataset)}"
        )
        log(
            "[SELECTION] best_checkpoint_split=official_val "
            "final_report_split=official_test test_evaluations=1"
        )
    log(
        f"[DATA] student_image=224 teacher_image=32 train_batch={args.batch_size} "
        f"eval_batch={args.eval_batch_size} num_workers={args.num_workers} "
        f"drop_last_train=True smoke={args.smoke} "
        f"split={split_description}"
    )
    return train_loader, test_loader, final_test_loader


def build_alg_loaders(
    args: argparse.Namespace,
    device: torch.device,
    timm: Any,
) -> tuple[Any, Any]:
    """Backward-compatible two-loader interface for historical protocols."""

    train_loader, selection_loader, _ = build_alg_loaders_with_final_test(
        args, device, timm
    )
    return train_loader, selection_loader


# Public names for the LG package.  The historical ``build_alg_*`` names stay
# available because the frozen Ours protocol imports them.
build_lg_loaders_with_final_test = build_alg_loaders_with_final_test
build_lg_loaders = build_alg_loaders


def build_native_teacher_audit_loader(
    args: argparse.Namespace,
    device: torch.device,
) -> Any:
    if args.dataset == "cifar100":
        dataset: Dataset[Any] = CIFAR100(
            root=args.data_dir,
            train=False,
            transform=official_test_transform(),
            download=False,
        )
    elif args.dataset == "flowers102":
        dataset: Dataset[Any] = Flowers102(
            root=args.data_dir,
            split="test",
            transform=official_test_transform(),
            download=False,
        )
    elif args.dataset == "chaoyang":
        from teachers.train_teacher_chaoyang import (
            ChaoyangDataset,
            resolve_dataset_root,
        )

        dataset_root = resolve_dataset_root(args.data_dir)
        dataset = ChaoyangDataset(
            dataset_root,
            "test",
            official_test_transform(),
        )
    elif args.dataset == "cub200":
        from teachers.cub200_dataset import CUB200Dataset, ensure_cub200

        dataset_root = ensure_cub200(args.data_dir)
        dataset = CUB200Dataset(
            dataset_root,
            split="test",
            transform=official_test_transform(),
        )
    else:
        raise ValueError(f"Unsupported teacher-audit dataset: {args.dataset}")
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
    student = timm.create_model(
        STUDENT_MODELS["deit_ti"],
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
    # The public LG DeiT initializes the classifier only after applying the
    # generic transformer initializer, then explicitly zeros weight and bias.
    head = getattr(student, "head", None)
    if not isinstance(head, torch.nn.Linear):
        raise RuntimeError("Expected DeiT-Ti to expose a linear `head` module")
    torch.nn.init.zeros_(head.weight)
    if head.bias is not None:
        torch.nn.init.zeros_(head.bias)
    return student


def official_lg_parameter_groups(
    student: torch.nn.Module,
    guidance: LocalityGuidance,
    *,
    lr: float,
    weight_decay: float,
    head_lr_ratio: float = 1.0,
) -> list[dict[str, Any]]:
    """Port the four official LG AdamW groups to the timm model layout."""

    groups: dict[str, list[torch.nn.Parameter]] = {
        "head_no_decay": [],
        "head_decay": [],
        "body_no_decay": [],
        "body_decay": [],
    }
    skip_tokens = ("cls_token", "pos_embed", "distill_token")
    named_parameters = [
        (f"student.{name}", parameter)
        for name, parameter in student.named_parameters()
    ] + [
        (f"feature_transforms.{name}", parameter)
        for name, parameter in guidance.named_parameters()
    ]
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        is_head = name.startswith("student.head.")
        no_decay = (
            parameter.ndim == 1
            or name.endswith(".bias")
            or any(token in name for token in skip_tokens)
        )
        key = (
            "head_no_decay"
            if is_head and no_decay
            else "head_decay"
            if is_head
            else "body_no_decay"
            if no_decay
            else "body_decay"
        )
        groups[key].append(parameter)

    head_lr = lr * head_lr_ratio
    return [
        {
            "name": "head_no_decay",
            "params": groups["head_no_decay"],
            "lr": head_lr,
            "weight_decay": 0.0,
        },
        {
            "name": "head_decay",
            "params": groups["head_decay"],
            "lr": head_lr,
            "weight_decay": weight_decay,
        },
        {
            "name": "body_no_decay",
            "params": groups["body_no_decay"],
            "lr": lr,
            "weight_decay": 0.0,
        },
        {
            "name": "body_decay",
            "params": groups["body_decay"],
            "lr": lr,
            "weight_decay": weight_decay,
        },
    ]


def create_official_lg_optimizer(
    student: torch.nn.Module,
    guidance: LocalityGuidance,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        official_lg_parameter_groups(
            student,
            guidance,
            lr=lr,
            weight_decay=weight_decay,
        ),
        lr=lr,
        weight_decay=weight_decay,
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
    controller: AdaptiveGuidanceController | StaticGuidanceController,
) -> dict[str, Any]:
    method = args.method
    return {
        "model": student.state_dict(),
        "model_state": student.state_dict(),
        "lg_guidance": guidance.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
        "best_accuracy": best_accuracy,
        "method": method,
        "dataset": args.dataset,
        "student": "deit_ti",
        "timm_model": STUDENT_MODELS["deit_ti"],
        "num_classes": NUM_CLASSES[args.dataset],
        "teacher": teacher_spec,
        "controller": controller.state_dict(),
        "official_lg_commit": OFFICIAL_LG_COMMIT,
        "alg_paper_doi": ALG_PAPER_DOI if method == "ALG" else None,
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
    controller: AdaptiveGuidanceController | StaticGuidanceController,
    teacher_native_top1: float,
    teacher_shared_top1: float,
    final_test_accuracy: float | None = None,
) -> None:
    average_epoch = sum(epoch_times) / max(1, len(epoch_times))
    method = args.method
    reference = REFERENCE_TOP1[args.dataset]
    official_three_way = (
        args.dataset == "flowers102"
        and args.flowers_split_policy == "official_three_way"
    )
    reported_accuracy = (
        final_test_accuracy if final_test_accuracy is not None else best_accuracy
    )
    summary = {
        "status": "complete" if latest_epoch == args.student_epochs else "running",
        "method": method,
        "dataset": args.dataset,
        "student": "deit_ti",
        "teacher": teacher_spec,
        "objective": (
            "CE + beta * LG while ALG paper controller is active"
            if method == "ALG"
            else "CE + beta * LG for every epoch"
        ),
        "official_lg_commit": OFFICIAL_LG_COMMIT,
        "alg_paper_doi": ALG_PAPER_DOI if method == "ALG" else None,
        "controller": controller.state_dict(),
        "teacher_native_top1": teacher_native_top1,
        "teacher_shared_view_top1": teacher_shared_top1,
        "student_epochs": args.student_epochs,
        "latest_epoch": latest_epoch,
        "flowers_split_policy": args.flowers_split_policy,
        "checkpoint_selection_split": "val" if official_three_way else "test",
        "best_top1": reported_accuracy,
        "latest_top1": latest_accuracy,
        "selection_best_top1": best_accuracy,
        "latest_selection_top1": latest_accuracy,
        "best_validation_top1": best_accuracy if official_three_way else None,
        "final_test_top1": final_test_accuracy,
        "final_test_evaluations": 1 if final_test_accuracy is not None else 0,
        "paper_alg_top1": reference["alg"],
        "gap_to_paper_alg_pp": (
            reported_accuracy - reference["alg"]
            if method == "ALG" and reference["alg"] is not None
            else None
        ),
        "gap_to_paper_lg_pp": (
            reported_accuracy - reference["lg"]
            if method == "LG" and reference["lg"] is not None
            else None
        ),
        "paper_lg_top1": (
            reference["alg_lg"] if method == "ALG" else reference["lg"]
        ),
        "paper_baseline_top1": (
            reference["alg_baseline"]
            if method == "ALG"
            else reference["baseline"]
        ),
        "paper_guidance_stop_epoch": PAPER_GUIDANCE_STOP_EPOCH[args.dataset],
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


def main(method: str = "ALG") -> None:
    if method not in {"LG", "ALG"}:
        raise ValueError(f"Unsupported locality-guidance method: {method}")
    install_signal_handlers()
    args = parse_args()
    args.method = method
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
    log(
        f"{'ADAPTIVE ' if method == 'ALG' else ''}"
        "LOCALITY GUIDANCE / RESNET56 -> DEIT-TI"
    )
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
    log(
        "[AUGMENT] color_jitter=0.4 auto_augment=rand-m9-mstd0.5-inc1 "
        "random_erasing=0.25/pixel/1 interpolation=bicubic "
        "normalization=ImageNet drop_last_train=True"
    )
    if method == "ALG":
        log(
            f"[ALG] loss=CE+beta*LG_while_controller_active "
            f"beta={args.beta} tau={args.alg_threshold} "
            f"smoothing_window={args.alg_smoothing_window} "
            f"controller_warm_up={args.alg_warmup_epochs} "
            f"stop_comparison={args.alg_stop_comparison} "
            f"derivative_mode={args.alg_derivative_mode} "
            "descent_guard=False observed_signal=complete_lg_loss"
        )
    else:
        log(
            f"[LG] loss=CE+beta*LG_all_epochs beta={args.beta} "
            "guidance_schedule=static"
        )
    log(
        "[LG] teacher_stages=(0,1,2) student_blocks=(0,6,11) "
        "projection=1x1 grid=larger_of_teacher_student interpolation=bilinear "
        "stage_loss=elementwise_MSE sum_stages=True"
    )
    reference = REFERENCE_TOP1[args.dataset]
    reference_names = (
        ("alg", "alg_lg", "alg_baseline")
        if method == "ALG"
        else ("lg", "baseline")
    )
    reference_text = " ".join(
        f"{name.upper()}="
        + ("n/a" if reference[name] is None else f"{reference[name]:.2f}%")
        for name in reference_names
    )
    log(
        f"[REFERENCE] dataset={args.dataset} {reference_text} "
        f"paper_guidance_stop_epoch={PAPER_GUIDANCE_STOP_EPOCH[args.dataset]}"
    )
    source_text = f"LG_repo_commit={OFFICIAL_LG_COMMIT}"
    if method == "ALG":
        source_text = f"ALG_DOI={ALG_PAPER_DOI} {source_text}"
    log(f"[SOURCE] {source_text}")

    (
        train_loader,
        test_loader,
        final_test_loader,
    ) = build_alg_loaders_with_final_test(args, device, timm)
    native_loader = build_native_teacher_audit_loader(args, device)
    teacher, teacher_payload, teacher_spec = load_teacher(
        args.dataset,
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
        NUM_CLASSES[args.dataset],
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
    if tuple(logits_probe.shape) != (2, NUM_CLASSES[args.dataset]):
        raise RuntimeError(f"Unexpected logits: {tuple(logits_probe.shape)}")
    if not bool(torch.isfinite(lg_probe)):
        raise RuntimeError(f"Non-finite {method} probe loss")

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

    optimizer = create_official_lg_optimizer(
        student,
        guidance,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    group_description = ",".join(
        f"{group['name']}:{sum(parameter.numel() for parameter in group['params'])}"
        for group in optimizer.param_groups
    )
    log(
        "[OPTIMIZER] official_lg_groups="
        f"{group_description} no_decay=1d,bias,cls_token,pos_embed"
    )
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
    scaler = create_grad_scaler(amp_enabled)
    controller: AdaptiveGuidanceController | StaticGuidanceController
    if method == "ALG":
        controller = AdaptiveGuidanceController(
            beta=args.beta,
            threshold=args.alg_threshold,
            smoothing_window=args.alg_smoothing_window,
            warm_up=args.alg_warmup_epochs,
            stop_comparison=args.alg_stop_comparison,
            derivative_mode=args.alg_derivative_mode,
        )
    else:
        controller = StaticGuidanceController(beta=args.beta)
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
            f"[{method}][{epoch:03d}/{args.student_epochs:03d}] loss={loss:.4f} "
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
    final_test_accuracy: float | None = None
    if final_test_loader is not None:
        best_payload = torch.load(
            best_checkpoint,
            map_location=device,
            weights_only=False,
        )
        student.load_state_dict(best_payload["model"])
        final_test_accuracy = evaluate(
            student, final_test_loader, device, amp_enabled
        )
        elapsed = time.time() - training_start
        best_payload["selection_split"] = "val"
        best_payload["selection_best_accuracy"] = best_accuracy
        best_payload["final_test_accuracy"] = final_test_accuracy
        atomic_torch_save(best_payload, best_checkpoint)
        write_summary(
            summary_path,
            args=args,
            teacher_spec=teacher_spec,
            latest_epoch=args.student_epochs,
            best_accuracy=best_accuracy,
            latest_accuracy=latest_accuracy,
            epoch_times=epoch_times,
            elapsed=elapsed,
            controller=controller,
            teacher_native_top1=teacher_native_top1,
            teacher_shared_top1=teacher_shared_top1,
            final_test_accuracy=final_test_accuracy,
        )
    log("=" * 72)
    if final_test_accuracy is None:
        reported_accuracy = best_accuracy
        reference_value = reference[method.lower()]
        comparison = (
            "paper_top1=n/a gap_to_paper=n/a"
            if reference_value is None
            else (
                f"paper_{method.lower()}_top1={reference_value:.2f}% "
                f"gap_to_paper={best_accuracy - reference_value:+.2f}pp"
            )
        )
        log(
            f"[FINAL_RESULT] {method.lower()}_best_top1={best_accuracy:.2f}% "
            f"{comparison}"
        )
    else:
        reported_accuracy = final_test_accuracy
        reference_value = reference[method.lower()]
        comparison = (
            "paper_top1=n/a test_gap_to_paper=n/a"
            if reference_value is None
            else (
                f"paper_{method.lower()}_top1={reference_value:.2f}% "
                f"test_gap_to_paper="
                f"{final_test_accuracy - reference_value:+.2f}pp"
            )
        )
        log(
            f"[FINAL_RESULT] {method.lower()}_best_val_top1={best_accuracy:.2f}% "
            f"{method.lower()}_final_test_top1={final_test_accuracy:.2f}% "
            f"{comparison}"
        )
        log(
            "[FINAL_TEST] selected_checkpoint=student_best.pt "
            "selection_split=official_val evaluation_split=official_test "
            "evaluation_count=1"
        )
    lg_reference = (
        reference["alg_lg"] if method == "ALG" else reference["lg"]
    )
    if lg_reference is not None:
        log(
            f"[FINAL_RESULT] paper_lg_top1={lg_reference:.2f}% "
            f"gain_over_paper_lg={reported_accuracy - lg_reference:+.2f}pp"
        )
    log(
        f"[BETA_FINAL] observed_stop_epoch={controller.stop_epoch} "
        f"paper_stop_epoch={PAPER_GUIDANCE_STOP_EPOCH[args.dataset]} "
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
    log(
        f"[DONE] {method} training completed successfully; "
        "resources may be released."
    )


def cli_main(method: str = "ALG") -> None:
    try:
        main(method)
    except Exception as error:
        log("=" * 72)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log(f"[FATAL] {method} training did not complete.")
        raise


if __name__ == "__main__":
    cli_main("LG")
