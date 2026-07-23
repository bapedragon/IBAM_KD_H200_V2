"""Thin training adapter that runs Ours V2 on the frozen Ours V1 protocol.

The data, teacher, student, optimizer, controller, loss weighting, checkpoint
logic, and CLI remain in :mod:`methods.Ours.core`.  This adapter replaces only
the auxiliary Ours module inside the current Python process and annotates every
checkpoint/summary so V1 and V2 artifacts cannot be confused.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch.nn as nn

from methods.Ours import core as base_core
from methods.OursV2.ours import Ours


ModelFactory = Callable[..., nn.Module]
_ORIGINAL_CHECKPOINT_PAYLOAD = base_core.checkpoint_payload
_ORIGINAL_FINALIZE_ARGS = base_core.finalize_args
_ORIGINAL_WRITE_SUMMARY = base_core.write_summary
_ACTIVE_VARIANT = "relative_position_v1"


def variant_metadata(variant: str) -> dict[str, Any]:
    common = {
        "package": "methods.OursV2",
        "variant": variant,
        "base_training_protocol": "methods.Ours.core",
        "original_ours_untouched": True,
    }
    if variant == "relative_position_v1":
        return {
            **common,
            "fusion": "conv1x1_global_cross_attention_plus_2d_relative_bias",
            "relative_position_bias": (
                "zero_initialized_learned_full_2d_displacement_table_per_head"
            ),
        }
    if variant == "grid_permutation_v1":
        return {
            **common,
            "fusion": "position_aware_attention_with_joint_kv_grid_permutation",
            "permutation_scope": "fusion_teacher_kv_only",
            "mse_targets": "unpermuted",
        }
    if variant == "token_space_v1":
        return {
            **common,
            "fusion": "linear_qkv_global_token_attention",
            "relative_position_bias": "disabled",
        }
    raise ValueError(f"Unsupported Ours V2 variant: {variant}")


def _checkpoint_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _ORIGINAL_CHECKPOINT_PAYLOAD(*args, **kwargs)
    payload["method"] = "OursV2"
    payload["ours_v2"] = variant_metadata(_ACTIVE_VARIANT)
    payload["base_ours_source_sha256"] = payload.pop(
        "source_snippet_sha256",
        None,
    )
    return payload


def _write_summary(*args: Any, **kwargs: Any) -> None:
    _ORIGINAL_WRITE_SUMMARY(*args, **kwargs)
    path_value = args[0] if args else kwargs["path"]
    path = Path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["method"] = "OursV2"
    payload["ours_v2"] = variant_metadata(_ACTIVE_VARIANT)
    payload["base_ours_source_sha256"] = payload.pop(
        "source_snippet_sha256",
        None,
    )
    temporary = path.with_suffix(path.suffix + ".ours_v2.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _finalize_args(args: Any) -> None:
    automatic_run_name = args.run_name is None
    _ORIGINAL_FINALIZE_ARGS(args)
    if automatic_run_name:
        args.run_name = args.run_name.replace(
            "ours_",
            f"ours_v2_{_ACTIVE_VARIANT}_",
            1,
        )


def cli_main(
    *,
    model_factory: ModelFactory = Ours,
    variant: str = "relative_position_v1",
) -> None:
    """Run one isolated Ours V2 variant through the established trainer."""

    global _ACTIVE_VARIANT
    variant_metadata(variant)
    _ACTIVE_VARIANT = variant
    base_core.Ours = model_factory  # type: ignore[assignment]
    base_core.checkpoint_payload = _checkpoint_payload
    base_core.finalize_args = _finalize_args
    base_core.write_summary = _write_summary
    print(
        "[OURS_V2] "
        f"variant={variant} original_methods_Ours_untouched=True "
        f"metadata={variant_metadata(variant)}",
        flush=True,
    )
    base_core.cli_main()
