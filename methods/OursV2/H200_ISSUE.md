# Ours V2 CIFAR-100 controls H200 Issue

Submit this timing Issue before requesting either 300-epoch control.  It runs
the two controls sequentially in the fixed order:

```text
1. grid_permutation_v1
2. token_space_v1
```

The timing run uses the complete CIFAR-100 train/test datasets for two epochs
per control while retaining the 300-epoch cosine horizon.  Its only purposes
are to catch runtime errors, verify the exact V2 variants, and estimate the two
300-epoch durations.  Timing accuracy is not a research result.

## GitHub Issue values

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CIFAR-100 Ours V2 grid-permutation token-space timing run` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/OursV2/run_cifar100_controls.py --timing-run --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

No additional install command is needed.  The repository pins
`timm==1.0.27`, and the training bootstrap installs it when necessary.

## Locked experiment contract

Both controls use CIFAR-100, the same fixed ResNet56 teacher, DeiT-Ti student,
300-epoch horizon, train/eval batch `64/200`, AdamW `5e-4`, minimum LR
`5e-6`, weight decay `0.05`, 20-epoch LR/controller warm-up, FP32, seed 1,
public-LG augmentation, and the same `32/16/14` stage grids.

- `grid_permutation_v1`: K and V receive the **same** fixed spatial
  permutation.  The learned 2D relative-position bias remains active and both
  MSE targets remain unpermuted.
- `token_space_v1`: flattened BNC tokens use Linear Q/K/V and content-only
  global attention; 2D relative-position bias is disabled.

The original `methods/Ours` package and every historical Ours checkpoint are
outside this request.

## Required success evidence

The H200 log must contain all of the following:

```text
[SEQUENCE][1/2] START variant=grid_permutation_v1
[OURS_V2_GRID_PERMUTATION] enabled=True ... scope=fusion_teacher_KV_only ...
[FEATURE_CHECK] ... stage_targets=[(2, 16, 32, 32), (2, 32, 16, 16), (2, 64, 14, 14)] ...
[SEQUENCE][1/2] DONE variant=grid_permutation_v1 estimated_300ep=...

[SEQUENCE][2/2] START variant=token_space_v1
[OURS_V2_TOKEN_SPACE] enabled=True ... relative_position_bias=False
[FEATURE_CHECK] ... stage_targets=[(2, 16, 32, 32), (2, 32, 16, 16), (2, 64, 14, 14)] ...
[SEQUENCE][2/2] DONE variant=token_space_v1 estimated_300ep=...

[FINAL_ESTIMATE][grid_permutation_v1] ... estimated_300ep=...
[FINAL_ESTIMATE][token_space_v1] ... estimated_300ep=...
[SEQUENCE_DONE] completed_tasks=2/2 estimated_combined_300ep=...
[POD_LIMIT_CHECK] status=PASS|FAIL ...
[DONE] Ours V2 CIFAR-100 control sequence completed successfully.
```

There must be no `[FATAL]`, Python traceback, CUDA OOM, non-finite loss, or
teacher-native audit failure.

The runner also validates that each child summary has:

```text
method = OursV2
ours_v2.variant = grid_permutation_v1 | token_space_v1
```

## Decision after timing

- If `[POD_LIMIT_CHECK] status=PASS`, the two 300-epoch controls may be packed
  into one follow-up Issue.
- If it reports `FAIL`, submit the two controls as separate full-run Issues.
- Do not use timing-run checkpoints or Top-1 values in the paper.

Timing artifacts are written under:

```text
/tmp/ours_v2_cifar100_controls_timing/
├── OursV2/cifar100/grid_permutation_v1/
├── OursV2/cifar100/token_space_v1/
├── sequence_status.json
└── sequence_summary.json
```

Only the Issue log and the printed duration estimates are needed to decide the
full-run packing.
