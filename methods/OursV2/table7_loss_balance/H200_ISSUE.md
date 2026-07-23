# Ours V2 CIFAR-100 lambda=0 full-run Issue

## GitHub Issue values

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CIFAR-100 Ours V2 Table 7 lambda 0 full run` |
| 사용자 ID | `kau-aimslab` |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/OursV2/table7_loss_balance/train_cifar100.py --lambda-value 0 --num-workers 4 --run-name table7_ours_v2_lambda_0_cifar100_300ep_seed1 --output-dir /app/output` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

## Request description

Run the Table 7 alignment-only control on CIFAR-100 for 300 epochs. Keep the
base Ours V2 `relative_position_v1` architecture and the complete locked
training protocol unchanged. Change only the convex loss-balance parameter
from the reference `lambda=0.5` to `lambda=0`, so the feature loss becomes:

```text
L_feature = L_align
```

This run is separate from the Table 4 `grid_permutation_v1` and
`token_space_v1` controls. Its valid comparison target is a completed base
Ours V2 `relative_position_v1`, `lambda=0.5` run under the same protocol.

Accept the result only if the log and summary show:

```text
[OURS_V2_TABLE7_CONTROL] variant=relative_position_v1 reference_lambda=0.5 only_change=lambda lambda=0
[OURS_V2_TABLE7_LOSS] feature_loss=0*L_fuse+1*L_align ...
[OURS_V2] variant=relative_position_v1 ...
[OURS] ... lambda=0.0
method = OursV2
ours_v2.variant = relative_position_v1
args.fusion_ratio = 0.0
student_epochs = 300
```

There must be no `[FATAL]`, traceback, CUDA OOM, non-finite loss, or failed
teacher audit.
