# CUB-200 LG / ALG / Ours H200 Issues

The timing Issue must run first. It trains one two-epoch scratch teacher and
then measures official LG, canonical paper ALG, and the unchanged Ours
protocol against the exact same generated teacher manifest.

## 1. Four-task timing Issue

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CUB-200 공용 ResNet56 teacher + LG + ALG + Ours timing run` |
| 개인 계정 사용자 ID | `bapedragon` |
| 연구실 계정 사용자 ID | `kau-aimslab` |
| 제출 계정 | 위 두 계정 중 실제 제출하는 계정 하나 |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_cub200_lg_alg_ours.py --timing-run --num-workers 4 --output-dir /app/output/cub200_lg_alg_ours_timing_seed1` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Pass evidence:

```text
[PROTOCOL_LOCK] LG=official ALG=paper_on_official_LG Ours=unchanged
[FAIRNESS] all students consume the exact same teacher manifest/checkpoint
[SEQUENCE_DONE] completed_tasks=4/4
[POD_LIMIT_CHECK] status=PASS
```

The first run can download the official CUB archive. The dataset helper checks
the archive MD5, metadata consistency, official `5,994/5,794` split, 200
classes, and all image paths.

## 2. Combined full Issue

Submit this only if the timing Issue ends in `status=PASS`.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CUB-200 공용 ResNet56 teacher + LG + ALG + Ours 300-epoch training` |
| 개인 계정 사용자 ID | `bapedragon` |
| 연구실 계정 사용자 ID | `kau-aimslab` |
| 제출 계정 | 위 두 계정 중 실제 제출하는 계정 하나 |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_cub200_lg_alg_ours.py --full-run --num-workers 4 --output-dir /app/output/cub200_lg_alg_ours_300ep_seed1` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Successful completion must contain:

```text
[DONE] CUB-200 teacher training completed successfully.
[DONE] LG training completed successfully; resources may be released.
[DONE] ALG training completed successfully; resources may be released.
[DONE] Ours training completed successfully; resources may be released.
[SEQUENCE_DONE] completed_tasks=4/4
```

## 3. Split full Issues if the combined estimate fails

Persist the completed 300-epoch teacher directory, including
`manifest.json`, for all three student Issues. Never use the two-epoch timing
teacher for a full student run.

Shared fields for every split Issue:

| Field | Value |
|---|---|
| 개인 계정 사용자 ID | `bapedragon` |
| 연구실 계정 사용자 ID | `kau-aimslab` |
| 제출 계정 | 위 두 계정 중 실제 제출하는 계정 하나 |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Teacher:

```text
Title: [Request]: 박철현 CUB-200 ResNet56 32x32 scratch teacher 300-epoch training
python teachers/train_teacher_cub200.py --data-dir ./data/cub200 \
  --output-dir /app/output/cub200_teacher --num-workers 4 \
  --run-name teacher_cub200_resnet56_32_scratch_300ep_seed1
```

LG, after mounting the full teacher directory at
`/app/input/cub200_teacher`:

```text
Title: [Request]: 박철현 CUB-200 official LG DeiT-Ti 300-epoch training
python methods/LG/cub200/train.py --data-dir ./data/cub200 \
  --teacher-root /app/input/cub200_teacher --output-dir /app/output \
  --run-name lg_cub200_deit_ti_300ep_seed1 --num-workers 4
```

ALG:

```text
Title: [Request]: 박철현 CUB-200 canonical ALG DeiT-Ti 300-epoch training
python methods/ALG/cub200/train.py --data-dir ./data/cub200 \
  --teacher-root /app/input/cub200_teacher --output-dir /app/output \
  --run-name alg_cub200_deit_ti_300ep_seed1 --num-workers 4
```

Ours:

```text
Title: [Request]: 박철현 CUB-200 Ours DeiT-Ti 300-epoch training
python methods/Ours/cub200/train.py --data-dir ./data/cub200 \
  --teacher-root /app/input/cub200_teacher --output-dir /app/output \
  --run-name ours_cub200_deit_ti_300ep_seed1 --num-workers 4
```
