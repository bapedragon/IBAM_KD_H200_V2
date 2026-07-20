#!/usr/bin/env python3
"""Train DeiT-Ti with official-code-based ReviewKD."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.KD.core import (
    NUM_CLASSES,
    STUDENT_MODELS,
    VANILLA_TOP1,
    autocast_context,
    atomic_torch_save,
    build_loaders,
    count_parameters,
    create_grad_scaler,
    create_scheduler,
    create_student,
    ensure_timm,
    evaluate,
    format_duration,
    install_signal_handlers,
    log,
    public_args,
    seed_everything,
    student_view_to_teacher_view,
    top1_correct,
)
from methods.ReviewKD.official_reviewkd import (
    ReviewKDAdapter,
    hierarchical_context_loss,
)
from teachers.verify_checkpoints import DEFAULT_CHECKPOINT_ROOT, load_teacher


OFFICIAL_REPOSITORY = "https://github.com/dvlab-research/ReviewKD"
OFFICIAL_COMMIT = "cede6ea6387ae9b6127de0e561507177bf19c11e"
TEACHER_CHANNELS = (16, 32, 64)
STUDENT_CHANNELS = (192, 192, 192)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(NUM_CLASSES), default="cifar100")
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
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-train-samples", type=int, default=1024)
    parser.add_argument("--smoke-test-samples", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--review-weight", type=float, default=0.6)
    parser.add_argument("--review-warmup-epochs", type=int, default=20)
    parser.add_argument(
        "--student-blocks",
        type=int,
        nargs=3,
        default=(3, 7, 11),
        metavar=("SHALLOW", "MIDDLE", "DEEP"),
    )
    parser.add_argument("--feature-grid", type=int, default=14)
    parser.add_argument("--mid-channels", type=int, default=192)
    parser.add_argument(
        "--amp",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    return parser.parse_args()


def finalize_args(args: argparse.Namespace) -> None:
    args.student_blocks = tuple(args.student_blocks)
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
        args.run_name = f"reviewkd_{args.dataset}_{args.student}_{suffix}"

    for field in (
        "student_epochs",
        "batch_size",
        "image_size",
        "smoke_train_samples",
        "smoke_test_samples",
        "lr",
        "feature_grid",
        "mid_channels",
    ):
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.warmup_epochs < 0 or args.review_warmup_epochs < 0:
        raise ValueError("Warm-up epochs must be non-negative")
    if args.review_weight < 0:
        raise ValueError("--review-weight must be non-negative")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.image_size != 224:
        raise ValueError("The fixed dataset protocols require --image-size 224")
    if args.feature_grid != 14:
        raise ValueError("DeiT-Ti patch features require --feature-grid 14")
    if (
        tuple(sorted(args.student_blocks)) != args.student_blocks
        or len(set(args.student_blocks)) != len(args.student_blocks)
    ):
        raise ValueError("--student-blocks must be strictly increasing")
    if args.student_blocks[0] < 0 or args.student_blocks[-1] >= 12:
        raise ValueError("DeiT-Ti block indexes must be in [0, 11]")


def forward_teacher_features(
    teacher: torch.nn.Module,
    images: torch.Tensor,
    feature_grid: int,
) -> list[torch.Tensor]:
    stages = teacher.forward_features(images)
    return [
        F.interpolate(
            feature,
            size=(feature_grid, feature_grid),
            mode="bilinear",
            align_corners=False,
        )
        for feature in stages
    ]


def forward_student_features(
    student: torch.nn.Module,
    images: torch.Tensor,
    student_blocks: tuple[int, int, int],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    final_tokens, intermediate_features = student.forward_intermediates(
        images,
        indices=list(student_blocks),
        norm=False,
        output_fmt="NCHW",
    )
    logits = student.forward_head(final_tokens)
    return list(intermediate_features), logits


def review_factor(epoch_index: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch_index / warmup_epochs)


def checkpoint_payload(
    student: torch.nn.Module,
    adapter: ReviewKDAdapter,
    epoch: int,
    accuracy: float,
    best_accuracy: float,
    args: argparse.Namespace,
    teacher_spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": student.state_dict(),
        "review_adapter": adapter.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
        "best_accuracy": best_accuracy,
        "method": "ReviewKD",
        "student": args.student,
        "timm_model": STUDENT_MODELS[args.student],
        "dataset": args.dataset,
        "num_classes": NUM_CLASSES[args.dataset],
        "teacher": teacher_spec,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
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
    summary = {
        "status": "complete" if latest_epoch == args.student_epochs else "running",
        "method": "ReviewKD",
        "dataset": args.dataset,
        "student": args.student,
        "timm_model": STUDENT_MODELS[args.student],
        "teacher": teacher_spec,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "student_epochs": args.student_epochs,
        "latest_epoch": latest_epoch,
        "best_top1": best_accuracy,
        "latest_top1": latest_accuracy,
        "vanilla_top1": VANILLA_TOP1[args.dataset][args.student],
        "gain_over_vanilla_pp": (
            best_accuracy - VANILLA_TOP1[args.dataset][args.student]
        ),
        "epoch_times": epoch_times,
        "avg_epoch_seconds": average_epoch,
        "planned_epochs": args.planned_epochs,
        "estimated_planned_seconds": average_epoch * args.planned_epochs,
        "estimated_planned_human": format_duration(
            average_epoch * args.planned_epochs
        ),
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
    log("OFFICIAL-CODE-BASED REVIEWKD / RESNET56 -> DEIT-TI")
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
        f"student_epochs={args.student_epochs} "
        f"planned_epochs={args.planned_epochs}"
    )
    log(
        f"[PROTOCOL] name={args.protocol_name} optimizer=AdamW "
        f"lr={args.lr} weight_decay={args.weight_decay} "
        f"warmup={args.warmup_epochs} cosine batch={args.batch_size} "
        f"image={args.image_size}"
    )
    log(
        "[INPUT] shared_geometry=True student=224x224(CIFAR-100 norm) "
        "teacher=bilinear_downsample_to_32x32(ImageNet norm)"
    )
    log(
        f"[REVIEWKD] loss=CE+ramp*{args.review_weight}*HCL "
        f"feature_warmup={args.review_warmup_epochs} no_logit_KL"
    )
    log(
        f"[OFFICIAL] repository={OFFICIAL_REPOSITORY} "
        f"commit={OFFICIAL_COMMIT}"
    )
    log(
        "[ADAPTER] teacher=post-activation stage1/2/3 bilinear_to_14x14 "
        f"student_blocks={args.student_blocks} token_grid=14x14 "
        f"channels={STUDENT_CHANNELS}->{TEACHER_CHANNELS} "
        f"abf_mid={args.mid_channels}"
    )

    train_loader, test_loader = build_loaders(args, device)
    teacher, teacher_payload, teacher_spec = load_teacher(
        args.dataset,
        device=device,
        checkpoint_root=args.teacher_root,
    )
    student = create_student(
        timm,
        args.student,
        NUM_CLASSES[args.dataset],
    ).to(device)
    adapter = ReviewKDAdapter(
        STUDENT_CHANNELS,
        TEACHER_CHANNELS,
        args.mid_channels,
    ).to(device)

    with torch.no_grad():
        probe = torch.zeros(2, 3, args.image_size, args.image_size, device=device)
        teacher_probe = forward_teacher_features(
            teacher,
            student_view_to_teacher_view(probe),
            args.feature_grid,
        )
        student_probe, logits_probe = forward_student_features(
            student,
            probe,
            args.student_blocks,
        )
        reviewed_probe = adapter(student_probe)
    expected_teacher_shapes = [
        (2, channels, args.feature_grid, args.feature_grid)
        for channels in TEACHER_CHANNELS
    ]
    if [tuple(feature.shape) for feature in teacher_probe] != expected_teacher_shapes:
        raise RuntimeError(
            "Unexpected teacher feature shapes: "
            f"{[tuple(feature.shape) for feature in teacher_probe]}"
        )
    if [tuple(feature.shape) for feature in student_probe] != [
        (2, 192, 14, 14)
    ] * 3:
        raise RuntimeError(
            "Unexpected student feature shapes: "
            f"{[tuple(feature.shape) for feature in student_probe]}"
        )
    if [tuple(feature.shape) for feature in reviewed_probe] != expected_teacher_shapes:
        raise RuntimeError(
            "Unexpected reviewed feature shapes: "
            f"{[tuple(feature.shape) for feature in reviewed_probe]}"
        )
    if tuple(logits_probe.shape) != (2, NUM_CLASSES[args.dataset]):
        raise RuntimeError(f"Unexpected student logits: {tuple(logits_probe.shape)}")

    log(
        f"[TEACHER] selected={teacher_spec['selected_kind']} "
        f"epoch={teacher_payload['epoch']} "
        f"top1={float(teacher_payload['accuracy']):.2f}% "
        f"sha256={teacher_spec['sha256']}"
    )
    log(
        f"[MODEL] teacher_params={count_parameters(teacher):,} "
        f"student_params={count_parameters(student):,} "
        f"review_trainable_params={count_parameters(adapter):,}"
    )
    log(
        f"[FEATURE_CHECK] teacher={expected_teacher_shapes} "
        f"student={[(2, 192, 14, 14)] * 3} "
        f"reviewed={expected_teacher_shapes}"
    )

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(adapter.parameters()),
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
        f"[STUDENT] optimizer=adamw lr={args.lr} "
        f"weight_decay={args.weight_decay} epochs={args.student_epochs} "
        f"effective_warmup={effective_warmup}"
    )

    best_accuracy = 0.0
    latest_accuracy = 0.0
    epoch_times: list[float] = []
    training_start = time.time()

    for epoch_index in range(args.student_epochs):
        epoch = epoch_index + 1
        epoch_start = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]
        factor = review_factor(epoch_index, args.review_warmup_epochs)
        student.train()
        adapter.train()
        teacher.eval()
        total_loss = 0.0
        total_ce = 0.0
        total_review = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad(), autocast_context(amp_enabled):
                teacher_features = forward_teacher_features(
                    teacher,
                    student_view_to_teacher_view(images),
                    args.feature_grid,
                )
            with autocast_context(amp_enabled):
                student_features, student_logits = forward_student_features(
                    student,
                    images,
                    args.student_blocks,
                )
                reviewed_features = adapter(student_features)
                classification_loss = F.cross_entropy(
                    student_logits,
                    targets,
                    label_smoothing=args.label_smoothing,
                )

            review_loss = hierarchical_context_loss(
                [feature.float() for feature in reviewed_features],
                [feature.float() for feature in teacher_features],
            )
            loss = (
                classification_loss.float()
                + factor * args.review_weight * review_loss
            )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_batch = targets.size(0)
            total += current_batch
            total_loss += float(loss.detach()) * current_batch
            total_ce += float(classification_loss.detach()) * current_batch
            total_review += float(review_loss.detach()) * current_batch
            correct += top1_correct(student_logits.detach(), targets)

        denominator = max(1, total)
        average_loss = total_loss / denominator
        average_ce = total_ce / denominator
        average_review = total_review / denominator
        train_accuracy = 100.0 * correct / denominator
        latest_accuracy = evaluate(student, test_loader, device, amp_enabled)
        epoch_seconds = time.time() - epoch_start
        epoch_times.append(epoch_seconds)
        previous_best = best_accuracy
        best_accuracy = max(best_accuracy, latest_accuracy)

        payload = checkpoint_payload(
            student,
            adapter,
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
            f"[REVIEWKD][{epoch:03d}/{args.student_epochs:03d}] "
            f"loss={average_loss:.4f} ce={average_ce:.4f} "
            f"hcl={average_review:.4f} ramp={factor:.3f} "
            f"train_acc={train_accuracy:.2f}% "
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
        f"[FINAL_RESULT] reviewkd_best_top1={best_accuracy:.2f}% "
        f"vanilla_top1={vanilla:.2f}% "
        f"gain_over_vanilla={best_accuracy - vanilla:+.2f}pp"
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
    log(
        "[DONE] ReviewKD training completed successfully; "
        "resources may be released."
    )


def cli_main() -> None:
    try:
        main()
    except Exception as error:
        log("=" * 72)
        log(f"[FATAL] {type(error).__name__}: {error}")
        traceback.print_exc()
        log("[FATAL] ReviewKD training did not complete.")
        raise


if __name__ == "__main__":
    cli_main()
