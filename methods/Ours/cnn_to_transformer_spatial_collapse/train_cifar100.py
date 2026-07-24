#!/usr/bin/env python3
"""Run the Ours V1 teacher-CNN to transformer-collapse experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.Ours import core
from methods.Ours.cnn_to_transformer_spatial_collapse.model import (
    CNNToTransformerSpatialCollapseOurs,
)


PROTOCOL_NAME = (
    "cifar100_deit_ti_ours_v1_cnn_to_transformer_spatial_collapse_v1"
)
PROTOCOL_DEFAULTS = (
    ("--protocol-name", PROTOCOL_NAME),
    ("--student-epochs", "300"),
    ("--batch-size", "64"),
    ("--eval-batch-size", "200"),
    ("--lr", "0.0005"),
    ("--min-lr", "0.000005"),
    ("--weight-decay", "0.05"),
    ("--warmup-epochs", "20"),
    ("--warmup-factor", "0.001"),
    ("--label-smoothing", "0.0"),
    ("--drop-path-rate", "0.1"),
    ("--seed", "1"),
    ("--base-protocol", "lg_official"),
    ("--teacher-image-size", "32"),
    ("--beta-schedule", "alg"),
    ("--beta-on", "2.5"),
    ("--alg-threshold", "-0.02"),
    ("--alg-smoothing-window", "50"),
    ("--alg-warmup-epochs", "20"),
    # The shared trainer requires this legacy argument. The isolated model
    # records it but always targets the student's single 14x14 patch grid.
    ("--grid-resize-mode", "larger"),
    ("--eval-resize-mode", "direct"),
)
VARIANT_METADATA = {
    "package": "methods.Ours.cnn_to_transformer_spatial_collapse",
    "variant": "cnn_to_transformer_spatial_collapse_v1",
    "base_architecture": "OursV1",
    "original_ours_v1_untouched": True,
    "projection_direction": "teacher_channels_16_32_64_to_student_dimension_192",
    "teacher_geometry": "each_stage_bilinear_to_student_patch_grid_14x14",
    "representation": "flattened_BND_196x192",
    "permutation": "none",
    "explicit_2d_position": "none",
    "loss_space": "transformer_token_representation",
}


def has_option(option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in sys.argv[1:]
    )


def install_protocol_defaults() -> None:
    if has_option("--dataset"):
        raise SystemExit("This entry point fixes --dataset cifar100.")
    sys.argv[1:1] = ["--dataset", "cifar100"]
    for option, value in reversed(PROTOCOL_DEFAULTS):
        if not has_option(option):
            sys.argv[1:1] = [option, value]


def install_variant() -> None:
    original_checkpoint_payload = core.checkpoint_payload
    original_finalize_args = core.finalize_args
    original_log = core.log
    original_write_summary = core.write_summary

    def checkpoint_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = original_checkpoint_payload(*args, **kwargs)
        payload["ours_variant"] = dict(VARIANT_METADATA)
        return payload

    def finalize_args(args: Any) -> None:
        automatic_run_name = args.run_name is None
        original_finalize_args(args)
        if automatic_run_name:
            args.run_name = args.run_name.replace(
                "ours_",
                "ours_v1_cnn_to_transformer_spatial_collapse_",
                1,
            )

    def write_summary(*args: Any, **kwargs: Any) -> None:
        original_write_summary(*args, **kwargs)
        path_value = args[0] if args else kwargs["path"]
        path = Path(path_value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["ours_variant"] = dict(VARIANT_METADATA)
        temporary = path.with_suffix(path.suffix + ".variant.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)

    def variant_log(message: str = "") -> None:
        if message.startswith("[OURS] student_blocks=all_12"):
            original_log(
                "[OURS_VARIANT] student_blocks=all_12 "
                "aggregation=learnable_uniform_init teacher_stages=1/2/3 "
                "teacher_resize=all_to_student_14x14 "
                "projection=Conv1x1_16/32/64_to_192 "
                "flatten=BND_196x192 attention=content_only "
                "permutation=none explicit_2d_position=none"
            )
            return
        if message.startswith("[REPRO_STATUS] Paper-confirmed:"):
            original_log(
                "[REPRO_STATUS] Base Ours V1 optimizer, controller, losses, "
                "aggregation, deformable student enhancement, and frozen "
                "teacher are retained. Variant change: reverse teacher CNN "
                "features into the common transformer representation before "
                "fusion and feature losses."
            )
            return
        original_log(message)

    core.Ours = CNNToTransformerSpatialCollapseOurs  # type: ignore[assignment]
    core.checkpoint_payload = checkpoint_payload
    core.finalize_args = finalize_args
    core.log = variant_log
    core.write_summary = write_summary


def main() -> None:
    install_protocol_defaults()
    install_variant()
    print(
        "[OURS_V1_CNN_TO_TRANSFORMER_SPATIAL_COLLAPSE] "
        f"protocol={PROTOCOL_NAME} metadata={VARIANT_METADATA}",
        flush=True,
    )
    core.cli_main()


if __name__ == "__main__":
    main()
