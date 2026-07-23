# Ours V2 CIFAR-100 lambda=0 -> 0.5 full-run Issue

## GitHub Issue values

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CIFAR-100 Ours V2 Table 7 lambda 0-0.5 300-epoch full run` |
| 사용자 ID | `kau-aimslab` |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/OursV2/table7_loss_balance/run_cifar100.py --full-run --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

## Verified timing evidence

The complete CIFAR-100 two-epoch timing audit completed both tasks without an
error:

```text
lambda=0:   estimated_300ep=3h 54m 54s
lambda=0.5: estimated_300ep=3h 56m 32s
combined:   estimated_300ep=7h 51m 26s
Pod limit:  PASS, headroom=2h 08m 34s
```

## Request description

Run the two Table 7 controls sequentially for 300 epochs each:

```text
1. Ours V2 relative_position_v1, lambda=0
2. Ours V2 relative_position_v1, lambda=0.5
```

Every architecture and training setting is identical. Only `lambda` changes:

```text
lambda=0:   L_feature = 0*L_fuse + 1*L_align
lambda=0.5: L_feature = 0.5*L_fuse + 0.5*L_align
```

Both runs use CIFAR-100, the same fixed ResNet56 teacher, DeiT-Ti student,
300-epoch cosine schedule, train/eval batch `64/200`, AdamW `5e-4`, minimum LR
`5e-6`, weight decay `0.05`, 20-epoch LR/controller warm-up, FP32, seed `1`,
larger-grid policy `32/16/14`, and the base Ours V2 learned head-specific 2D
relative-position bias.

This is the Table 7 loss-balance comparison. It is separate from the Table 4
`grid_permutation_v1` and `token_space_v1` controls.

## Required completion evidence

The final log must contain:

```text
[SEQUENCE] lambda_0 -> lambda_0p5
[COMPARISON_LOCK] variant=relative_position_v1 only_change=lambda
[SEQUENCE][1/2] DONE lambda=0 estimated_300ep=...
[SEQUENCE][2/2] DONE lambda=0.5 estimated_300ep=...
[SEQUENCE_DONE] completed_tasks=2/2 ...
[DONE] Ours V2 Table 7 lambda pair completed successfully.
```

Each completed summary must report:

```text
method = OursV2
ours_v2.variant = relative_position_v1
student_epochs = 300
args.fusion_ratio = 0.0 | 0.5
```

There must be no `[FATAL]`, traceback, CUDA OOM, non-finite loss, or failed
teacher audit. The two independent result directories are written below:

```text
/app/output/ours_v2_cifar100_table7_lambda_pair_300ep_seed1/
├── OursV2/cifar100/table7_loss_balance/lambda_0/
└── OursV2/cifar100/table7_loss_balance/lambda_0p5/
```
