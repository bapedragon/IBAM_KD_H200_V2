# CUB-200-2011 Ours H200 Issues

The CUB pipeline is new and includes both the required scratch teacher and the
Ours student. Submit the timing request first. Do not submit the combined full
request unless the timing log ends with `completed_tasks=2/2` and
`[POD_LIMIT_CHECK] status=PASS`.

## 1. Combined timing run

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CUB-200 ResNet56 teacher and Ours timing run` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/Ours/cub200/run_pipeline.py --timing-run --num-workers 4 --output-dir /app/output/cub200_ours_pipeline_timing_seed1` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Expected protocol evidence:

```text
[PROTOCOL_CHECK] status=PASS
[DATA] ... train_samples=5994 test_samples=5794 ...
[SEQUENCE_DONE] completed_tasks=2/2
[POD_LIMIT_CHECK] status=PASS|FAIL
```

The first execution may download the official 1.2 GB archive. The code checks
MD5 before extraction and validates all three metadata files, sample counts,
class count, split flags, and image paths.

## 2. Combined 300+300 epoch full run

Submit only after the timing request reports `PASS`.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CUB-200 ResNet56 teacher and Ours 300-epoch training` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/Ours/cub200/run_pipeline.py --full-run --num-workers 4 --output-dir /app/output/cub200_ours_pipeline_300ep_seed1` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Successful completion must include:

```text
[FINAL_RESULT] teacher_best_top1=...
[DONE] CUB-200 teacher training completed successfully.
[FINAL_RESULT] ours_best_top1=...
[DONE] Ours training completed successfully; resources may be released.
[SEQUENCE_DONE] completed_tasks=2/2
```

Artifacts to collect:

```text
/app/output/cub200_ours_pipeline_300ep_seed1/
├── teacher/teacher_cub200_resnet56_32_scratch_300ep_seed1/
│   ├── teacher_resnet56_cub200_32_best.pt
│   ├── teacher_resnet56_cub200_32_latest.pt
│   ├── manifest.json
│   ├── metrics.csv
│   └── summary.json
├── Ours/cub200/ours_cub200_deit_ti_300ep_seed1/
│   ├── student_best.pt
│   ├── student_latest.pt
│   └── summary.json
└── sequence_status.json
```

If timing reports `FAIL`, split the full work into two Issues. First run
`train_teacher.py` with a persistent output path, import its complete
teacher directory into the repository or another mounted persistent path, and
then run `train.py --teacher-root <DIRECTORY_CONTAINING_MANIFEST>`. Do not
start Ours from a two-epoch timing teacher.
