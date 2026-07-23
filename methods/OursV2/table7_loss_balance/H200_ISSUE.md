# Ours V2 CIFAR-100 lambda=0 -> 0.5 timing Issue

## GitHub Issue values

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CIFAR-100 Ours V2 Table 7 lambda 0-0.5 timing run` |
| 사용자 ID | `kau-aimslab` |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/OursV2/table7_loss_balance/run_cifar100.py --timing-run --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

## Request description

Run the two Table 7 controls sequentially on the complete CIFAR-100 train/test
datasets for two epochs each:

```text
1. Ours V2 relative_position_v1, lambda=0
2. Ours V2 relative_position_v1, lambda=0.5
```

Every training setting and model component is identical between the two runs.
Only the convex loss-balance parameter changes:

```text
lambda=0:   L_feature = 0*L_fuse + 1*L_align
lambda=0.5: L_feature = 0.5*L_fuse + 0.5*L_align
```

The purpose of this timing run is to verify both exact configurations, catch
runtime errors, estimate each 300-epoch duration, and decide whether the pair
fits the 10-hour Pod limit. Timing-run accuracy and checkpoints are not paper
results. This request is separate from the Table 4 `grid_permutation_v1` and
`token_space_v1` controls.

Accept the timing audit only if the log shows:

```text
[SEQUENCE] lambda_0 -> lambda_0p5
[COMPARISON_LOCK] variant=relative_position_v1 only_change=lambda

[SEQUENCE][1/2] START lambda=0 variant=relative_position_v1
[OURS_V2_TABLE7_CONTROL] variant=relative_position_v1 reference_lambda=0.5 only_change=lambda lambda=0
[OURS_V2_TABLE7_LOSS] feature_loss=0*L_fuse+1*L_align ...
[SEQUENCE][1/2] DONE lambda=0 estimated_300ep=...

[SEQUENCE][2/2] START lambda=0.5 variant=relative_position_v1
[OURS_V2_TABLE7_CONTROL] variant=relative_position_v1 reference_lambda=0.5 only_change=lambda lambda=0.5
[OURS_V2_TABLE7_LOSS] feature_loss=0.5*L_fuse+0.5*L_align ...
[SEQUENCE][2/2] DONE lambda=0.5 estimated_300ep=...

[FINAL_ESTIMATE][lambda=0] ...
[FINAL_ESTIMATE][lambda=0.5] ...
[SEQUENCE_DONE] completed_tasks=2/2 estimated_combined_300ep=...
[POD_LIMIT_CHECK] status=PASS|FAIL ...
[DONE] Ours V2 Table 7 lambda pair completed successfully.
```

There must be no `[FATAL]`, traceback, CUDA OOM, non-finite loss, or failed
teacher audit.

If `[POD_LIMIT_CHECK] status=PASS`, submit the paired 300-epoch full-run Issue.
If it reports `FAIL`, submit lambda `0` and `0.5` as separate full-run Issues.
