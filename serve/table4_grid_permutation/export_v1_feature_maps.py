#!/usr/bin/env python3
"""Export before/after feature maps from the completed V1 grid-permutation run."""

from __future__ import annotations

import argparse
import hashlib
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.transforms import InterpolationMode


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.KD.core import IMAGENET_MEAN, IMAGENET_STD, ensure_timm
from methods.Ours.core import (
    NUM_STUDENT_BLOCKS,
    STUDENT_CHANNELS,
    TEACHER_CHANNELS,
    create_ours_student,
    forward_student_features,
    forward_teacher_features,
)
from methods.Ours.table4_grid_permutation.model import GridPermutedOurs
from teachers.verify_checkpoints import DEFAULT_CHECKPOINT_ROOT, load_teacher


RUN_DIRECTORY = (
    REPOSITORY_ROOT
    / "results"
    / "Ours"
    / "cifar100"
    / "table4_grid_permuted_researcher_sync_v1_300ep_seed1_permseed1"
)
DEFAULT_CHECKPOINT = RUN_DIRECTORY / "student_best.pt"
DEFAULT_EXPORT_DIRECTORY = RUN_DIRECTORY / "feature_map_exports"
DEFAULT_SAMPLE_INDEX = 0
DEFAULT_PERMUTATION_SEED = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Completed V1 grid-permutation student checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data",
    )
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=DEFAULT_SAMPLE_INDEX,
        help="Zero-based CIFAR-100 official-test index.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination .pkl. The generated default name explicitly includes v1.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Prefer a repository-relative artifact path when one is available."""

    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(requested)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"V1 grid-permutation checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("method") != "Ours":
        raise RuntimeError(
            f"Expected method='Ours', found {payload.get('method')!r}"
        )
    checkpoint_args = payload.get("args", {})
    expected_protocol = "table4_grid_permuted_cifar100_researcher_sync_v1"
    if checkpoint_args.get("protocol_name") != expected_protocol:
        raise RuntimeError(
            "Checkpoint is not the completed V1 Table 4 run: "
            f"protocol={checkpoint_args.get('protocol_name')!r}"
        )
    if int(payload.get("epoch", -1)) != 298:
        raise RuntimeError(
            "Expected the selected 81.79% checkpoint from epoch 298, "
            f"found epoch={payload.get('epoch')!r}"
        )
    if abs(float(payload.get("accuracy", -1.0)) - 81.79) > 1e-8:
        raise RuntimeError(
            "Expected selected Top-1=81.79, "
            f"found {payload.get('accuracy')!r}"
        )
    return payload


def build_eval_transform() -> transforms.Compose:
    """Return the exact CIFAR-100 evaluation transform used by this V1 run."""

    return transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def restore_v1_models(
    checkpoint: dict[str, Any],
    device: torch.device,
    teacher_root: Path,
) -> tuple[torch.nn.Module, torch.nn.Module, GridPermutedOurs, dict[str, Any]]:
    timm = ensure_timm()
    checkpoint_args = checkpoint["args"]

    teacher, _, teacher_spec = load_teacher(
        "cifar100",
        device=device,
        checkpoint_root=teacher_root,
    )
    student = create_ours_student(
        timm,
        "deit_ti",
        100,
        float(checkpoint_args["drop_path_rate"]),
    ).to(device)
    student.load_state_dict(checkpoint["model"], strict=True)
    student.eval()

    ours = GridPermutedOurs(
        student_channels=STUDENT_CHANNELS,
        teacher_channels=TEACHER_CHANNELS,
        num_student_blocks=NUM_STUDENT_BLOCKS,
        num_heads=int(checkpoint_args["num_heads"]),
        spatial_kernel_size=int(checkpoint_args["deform_kernel_size"]),
        grid_resize_mode=str(checkpoint_args["grid_resize_mode"]),
        permutation_seed=DEFAULT_PERMUTATION_SEED,
    ).to(device)

    ours_state = checkpoint["ours"]
    for stage in range(len(TEACHER_CHANNELS)):
        name = f"table4_grid_permutation_stage_{stage}"
        permutation = ours_state[name].detach().clone().to(device=device)
        positions = int(permutation.numel())
        side = math.isqrt(positions)
        if side * side != positions:
            raise RuntimeError(
                f"Checkpoint permutation is not a square grid: {name}={positions}"
            )
        ours.register_buffer(name, permutation, persistent=True)
        ours._permutation_shapes[stage] = (side, side)

    ours.load_state_dict(ours_state, strict=True)
    ours.eval()
    return teacher, student, ours, teacher_spec


def inverse_permutation(permutation: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(
        permutation.numel(),
        device=permutation.device,
        dtype=permutation.dtype,
    )
    return inverse


def export_feature_maps(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    teacher_root = args.teacher_root.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)

    raw_dataset = CIFAR100(
        root=data_dir,
        train=False,
        transform=None,
        download=False,
    )
    if not 0 <= args.sample_index < len(raw_dataset):
        raise IndexError(
            f"--sample-index must be in [0, {len(raw_dataset) - 1}], "
            f"got {args.sample_index}"
        )
    original_image, label = raw_dataset[args.sample_index]
    class_name = raw_dataset.classes[label]
    student_view = build_eval_transform()(original_image).unsqueeze(0).to(device)

    teacher, student, ours, teacher_spec = restore_v1_models(
        checkpoint,
        device,
        teacher_root,
    )
    with torch.inference_mode():
        student_features, student_logits = forward_student_features(
            student,
            student_view,
        )
        raw_teacher_features = forward_teacher_features(
            teacher,
            student_view,
            dataset="cifar100",
            teacher_image_size=32,
            base_protocol="lg_official",
        )
        model_permuted_features = ours._resize_and_permute_teacher(
            student_features,
            raw_teacher_features,
        )

    student_grid = tuple(int(value) for value in student_features[0].shape[-2:])
    exported_stages: list[dict[str, Any]] = []
    for stage, (raw_teacher, actual_after) in enumerate(
        zip(raw_teacher_features, model_permuted_features, strict=True)
    ):
        target_size = (
            max(student_grid[0], int(raw_teacher.shape[-2])),
            max(student_grid[1], int(raw_teacher.shape[-1])),
        )
        before = raw_teacher
        if tuple(before.shape[-2:]) != target_size:
            before = F.interpolate(
                before,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        permutation = getattr(
            ours,
            f"table4_grid_permutation_stage_{stage}",
        )
        expected_after = before.flatten(2).index_select(2, permutation).reshape_as(
            before
        )
        torch.testing.assert_close(
            actual_after,
            expected_after,
            rtol=0.0,
            atol=0.0,
        )
        restored = actual_after.flatten(2).index_select(
            2,
            inverse_permutation(permutation),
        ).reshape_as(before)
        torch.testing.assert_close(restored, before, rtol=0.0, atol=0.0)

        exported_stages.append(
            {
                "stage": stage + 1,
                "raw_teacher_shape_bchw": tuple(int(x) for x in raw_teacher.shape),
                "exported_shape_chw": tuple(int(x) for x in before[0].shape),
                "grid_height": target_size[0],
                "grid_width": target_size[1],
                "stage_permutation_seed": DEFAULT_PERMUTATION_SEED + stage,
                "mapping": (
                    "after[:, output_position] = "
                    "before[:, permutation[output_position]]"
                ),
                "before_feature_map": (
                    before[0].detach().to(device="cpu", dtype=torch.float32).numpy()
                ),
                "after_feature_map": (
                    actual_after[0]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .numpy()
                ),
                "permutation": permutation.detach().cpu().numpy().astype(np.int64),
                "inverse_permutation": (
                    inverse_permutation(permutation)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                ),
                "exact_permutation_check_passed": True,
                "exact_inverse_recovery_check_passed": True,
            }
        )

    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_EXPORT_DIRECTORY / (
            "grid_permutation_v1_cifar100_test_idx"
            f"{args.sample_index:05d}_before_after_feature_maps.pkl"
        )
    output_path = output_path.expanduser().resolve()
    if output_path.suffix.lower() != ".pkl":
        raise ValueError(f"--output must end in .pkl: {output_path}")

    artifact = {
        "schema": "grid_permutation_before_after_feature_maps",
        "schema_version": 1,
        "experiment_version": "v1",
        "method": "Ours",
        "variant": "table4_grid_permutation_v1",
        "description": (
            "Same CIFAR-100 image and teacher activations immediately before "
            "and after the completed V1 run's fixed spatial permutation."
        ),
        "dataset": {
            "name": "cifar100",
            "split": "official_test",
            "sample_index": int(args.sample_index),
            "label_index": int(label),
            "class_name": class_name,
        },
        "source_checkpoint": {
            "path": portable_path(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "selected_epoch": int(checkpoint["epoch"]),
            "top1": float(checkpoint["accuracy"]),
            "protocol_name": checkpoint["args"]["protocol_name"],
            "run_name": checkpoint["args"]["run_name"],
        },
        "teacher_checkpoint": {
            "checkpoint": teacher_spec["checkpoint"],
            "sha256": teacher_spec["sha256"],
            "top1": float(teacher_spec["top1"]),
        },
        "preprocessing": {
            "student_view": (
                "bicubic direct resize to 224x224, ImageNet normalization"
            ),
            "teacher_view": (
                "bilinear resize of normalized student view to 32x32"
            ),
            "grid_resize_mode": "larger",
        },
        "permutation": {
            "run_seed": DEFAULT_PERMUTATION_SEED,
            "fixed_across_samples_and_epochs": True,
            "same_permuted_teacher_map_used_for": [
                "cross_attention_K",
                "cross_attention_V",
                "L_align_target",
                "L_fuse_target",
            ],
        },
        "input": {
            "original_rgb_uint8_hwc": np.asarray(original_image, dtype=np.uint8),
            "student_view_float32_chw": (
                student_view[0]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .numpy()
            ),
            "student_predicted_class": int(student_logits.argmax(dim=1).item()),
        },
        "stages": exported_stages,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as destination:
        pickle.dump(artifact, destination, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    output_path = export_feature_maps(args)
    print(f"[V1_FEATURE_EXPORT] path={output_path}", flush=True)
    print(f"[V1_FEATURE_EXPORT] sha256={sha256(output_path)}", flush=True)
    with output_path.open("rb") as source:
        artifact = pickle.load(source)
    for stage in artifact["stages"]:
        print(
            "[V1_FEATURE_EXPORT] "
            f"stage={stage['stage']} "
            f"shape={stage['exported_shape_chw']} "
            f"permutation_positions={len(stage['permutation'])} "
            "exact_check=True inverse_check=True",
            flush=True,
        )
    print("[V1_FEATURE_EXPORT] done", flush=True)


if __name__ == "__main__":
    main()
