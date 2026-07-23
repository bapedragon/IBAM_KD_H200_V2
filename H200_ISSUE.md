# H200 Issue request values

Use the timing request first when a pipeline is new. CIFAR-100 estimates its
300-epoch duration. The Flowers pipeline has already completed multiple full
runs, so its final 450-epoch request may be submitted directly.

The new CUB-200-2011 scratch-teacher + Ours timing/full request is maintained
with its code under
[`methods/Ours/cub200/H200_ISSUE.md`](methods/Ours/cub200/H200_ISSUE.md).

## 1. Timing run

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CIFAR-100 ResNet56 32x32 teacher timing run` |
| 사용자 ID | `bapedragon` |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python teachers/train_teacher_cifar100.py --timing-run --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

The script installs its pinned `timm==1.0.27` dependency automatically, so no
extra installation command is required in the execution field.

## Pure ALG batch ablation + CIFAR-locked Ours Chaoyang (one Issue)

This sequence runs four independent 300-epoch tasks: pure ALG Flowers batch
64, pure ALG Chaoyang batch 128, pure ALG Chaoyang batch 64, and Ours
Chaoyang batch 64 using the researcher-sync protocol fixed by the successful
CIFAR-100 run. Pure ALG uses the ALG equations/public-LG base; it does not use
the Ours controller or fusion module.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 ALG Flowers Chaoyang batch comparison and Ours Chaoyang training` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_alg_batch_ablation_ours_chaoyang.py --full-run --num-workers 4 --output-dir /app/output/alg_batch_ablation_ours_chaoyang_300ep_seed1` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Comparable completed runs indicate a total around three hours, comfortably
below the 600-minute Pod limit. For a fresh infrastructure check, replace
`--full-run` with `--timing-run` and omit `--output-dir`; the returned log will
print the new combined estimate and `[POD_LIMIT_CHECK]` status.

Successful completion prints four `[FINAL_BEST]` lines, `completed_tasks=4/4`,
`[POD_LIMIT_CHECK] status=PASS`, and the final sequence summary path. Each task
has its own `student_best.pt`, `student_latest.pt`, and `summary.json`. If a
later subprocess fails, earlier completed directories remain under
`/app/output` for collection.

## Flowers-102 official split Ours + ALG (current request)

This corrected protocol does **not** merge train and val. Both methods use
official train `1,020` for optimization, official val `1,020` for selecting
`student_best.pt`, and official test `6,149` only once after loading that best
checkpoint. Every other researcher-sync setting remains fixed: 300 epochs,
batch/eval batch `64/200`, AdamW `5e-4`, minimum LR `5e-6`, weight decay
`0.05`, warm-up 20, public LG augmentation, FP32, and seed 1.

### Timing run (submit first)

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Flowers-102 official-split Ours ALG timing run` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_flowers_official_split_ours_alg.py --timing-run --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

The log must show `train_samples=1020 val_samples=1020 test_samples=6149`,
`completed_tasks=2/2`, and `[POD_LIMIT_CHECK] status=PASS`.

### Full 300-epoch run

Submit after the timing run passes.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Flowers-102 official-split Ours ALG 300-epoch training` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_flowers_official_split_ours_alg.py --full-run --num-workers 4 --output-dir /app/output/flowers102_official_split_ours_alg_300ep_seed1` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

The final log reports separate `best_val_top1` and `final_test_top1` values.
The two method directories and the sequence status file are independent, so
one method cannot overwrite the other.

## 19. Researcher-sync Ours CIFAR/Flowers + ALG Flowers

This batch contains exactly three independent 300-epoch tasks in this order:
Ours CIFAR-100, Ours Flowers-102, and ALG Flowers-102. All three use the
researcher-sync base: train/eval batch `64/200`, AdamW `5e-4`, minimum LR
`5e-6`, weight decay `0.05`, warm-up 20 with factor `0.001`, drop path `0.1`,
label smoothing `0`, public LG strong augmentation, FP32, seed 1, 32-pixel
teacher input, 224-pixel student input, direct evaluation, and larger-grid
feature matching. Ours and ALG retain their distinct method losses.

### 19.1 Combined timing run (submit first)

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Ours CIFAR Flowers and ALG Flowers researcher-sync timing run` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_researcher_sync_ours_alg.py --timing-run --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

The final timing log must contain `completed_tasks=3/3` and
`[POD_LIMIT_CHECK] status=PASS` before a combined full run is submitted. A
`FAIL` means the jobs must be split across Issues. Timing artifacts are written
to `/tmp` and are not collected as research results.

### 19.2 Combined full run

Submit only after the timing estimate passes.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Ours CIFAR Flowers and ALG Flowers researcher-sync 300-epoch training` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_researcher_sync_ours_alg.py --full-run --num-workers 4 --output-dir /app/output/researcher_sync_ours_alg_300ep_seed1` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

The output tree is split by `Ours/cifar100`, `Ours/flowers102`, and
`ALG/flowers102`. Each task has independent best/latest checkpoints and a
summary. The root sequence status is updated after every task, so already
completed results remain available if a later subprocess fails. Historical
Flowers 200-epoch and Chaoyang 100-epoch results use explicit `historical`
file names and are never overwritten by this batch.

## Historical CIFAR-100 Ours + CRD + MGD timing (retired for Ours)

H200 build 451 completed this timing sequence successfully, but Ours used the
supplied-source `32/16/14` grid. Its Ours duration and result are historical;
do not use the sections below as a current paper-grid request. CRD/MGD are
unaffected. Re-time Ours with `stage_grid=teacher` before repacking it.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CIFAR-100 DeiT-Ti Ours CRD MGD timing run` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_cifar100_ours_crd_mgd.py --timing-run --output-dir /app/output/cifar100_ours_crd_mgd_timing_v2 --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Measured 300-epoch estimates were Ours `4h 08m 37s`, CRD `3h 15m 04s`, and
MGD `3h 03m 26s`. Their `10h 27m 07s` total exceeds the 600-minute Pod limit,
so do not submit all three in one full request.

### Historical CIFAR-100 Ours + CRD full request (do not resubmit as-is)

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CIFAR-100 DeiT-Ti Ours CRD full training` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_cifar100_ours_crd_mgd.py --full-run --methods Ours CRD --output-dir /app/output/cifar100_ours_crd_full_v2 --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Expected combined training time is approximately `7h 23m 41s`, leaving about
`2h 36m 19s` under the 600-minute Pod limit. Ours runs first, followed by CRD,
and both methods write independent result directories under `/app/output`.

### CIFAR-100 MGD full training

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CIFAR-100 DeiT-Ti MGD full training` |
| 사용자 ID | `kau-aimslab` (연구실 계정) **or** `bapedragon` (개인 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_cifar100_ours_crd_mgd.py --full-run --methods MGD --output-dir /app/output/cifar100_mgd_full_v2 --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Expected training time is approximately `3h 03m 26s`. Submit this after an
account becomes available. The current packing prioritizes Ours and CRD on the
idle account while the other account runs the previously submitted combined
job.

## 2. Full 300-epoch run

Submit this only after the timing log contains
`[PROTOCOL_CHECK] status=PASS` and `[DONE]`.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CIFAR-100 ResNet56 32x32 teacher training` |
| 사용자 ID | `bapedragon` |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python teachers/train_teacher_cifar100.py --output-dir /app/output --run-name teacher_resnet56_cifar100_32_lg_official_seed1 --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

## 3. Flowers-102 final-recipe timing run (account selectable, optional)

This request can run in parallel with the CIFAR-100 job because it uses a
different GitHub user allocation and writes only temporary timing artifacts.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Flowers-102 ResNet56 32x32 final teacher timing run` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python teachers/train_teacher_flowers.py --timing-run --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

## 4. Flowers-102 final full 450-epoch run (account selectable)

The previous Flowers run already verified the H200 data/model pipeline. The
new timing request is therefore optional; the full request may be submitted
directly after the updated commit is visible on GitHub.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Flowers-102 ResNet56 32x32 450-epoch teacher training` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python teachers/train_teacher_flowers.py --output-dir /app/output --run-name teacher_resnet56_flowers102_32_strongaug_450ep_seed1 --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

## 5. Chaoyang 32x32 teacher timing run (account selectable)

For Chaoyang requests, enter exactly one of the following IDs in the form:
`bapedragon` (personal account) or `kau-aimslab` (lab account). The command is
the same for both accounts because the dataset and output paths are shared.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Chaoyang ResNet56 32x32 teacher timing run` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python teachers/train_teacher_chaoyang.py --timing-run --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

The timing run reads `/app/data/chaoyang` and does not write to the collected
`/app/output` path.

## 6. Chaoyang full 300-epoch run (account selectable)

Submit after the timing log contains `[PROTOCOL_CHECK] status=PASS`, the exact
4,021/2,139 split counts, and `[DONE]`.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Chaoyang ResNet56 32x32 teacher training` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python teachers/train_teacher_chaoyang.py --output-dir /app/output --run-name teacher_resnet56_chaoyang_32_moderateaug_300ep_seed1 --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Do not change `/app/output` to `/app/outputs`; the H200 collector uses the
singular path.

## 7. CIFAR-100 KD + CRD + ReviewKD sequential timing run

This runs two full-dataset epochs for each method in the fixed order
`KD -> CRD -> ReviewKD`. Each method writes to a different directory. The
timing artifacts are collected so all three checkpoint/summary sets can be
inspected after the Pod exits.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CIFAR-100 DeiT-Ti KD CRD ReviewKD timing run` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_cifar100_three_methods.py --timing-run --output-dir /app/output/cifar100_three_methods_timing_v2 --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Successful completion must print all three per-method `[DONE]` markers and the
final sequence marker:

```text
[DONE] All three methods completed successfully; resources may be released.
```

## 8. CIFAR-100 full runs split safely across the 600-minute limit

The measured combined estimate was `9h 41m 37s`, leaving only about 18 minutes
before the 600-minute Pod limit. Run `KD + CRD` and `ReviewKD` as two separate
issues so normal runtime variance cannot discard all three results.

### 8.1 KD + CRD full run

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CIFAR-100 DeiT-Ti KD CRD full training` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_cifar100_three_methods.py --full-run --methods KD CRD --output-dir /app/output/cifar100_kd_crd_full_v2 --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Expected duration from timing: approximately `6h 23m 10s`.

### 8.2 ReviewKD full run

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CIFAR-100 DeiT-Ti ReviewKD full training` |
| 사용자 ID | `kau-aimslab` (연구실 계정) **or** `bapedragon` (개인 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_cifar100_three_methods.py --full-run --methods ReviewKD --output-dir /app/output/cifar100_reviewkd_full_v2 --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Expected duration from timing: approximately `3h 18m 27s`.

If a later method fails, the runner stops, writes the failing method and error
to `three_method_status.json`, and leaves every already completed method
directory untouched.

## 9. Flowers-102 + Chaoyang combined five-method timing run (recommended)

This single Issue measures all ten dataset-method combinations and is the most
convenient request for deciding the later full-run packing.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Flowers-102 Chaoyang DeiT-Ti 5-method timing run` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_flowers_chaoyang_timing.py --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

On success, it creates ten independent temporary method directories, two
per-dataset summaries, and `two_dataset_timing_summary.json`. They stay inside
the cloned working directory rather than the collected `/app/output` path;
the required estimates are all printed in the Issue log. If the second dataset
fails, the status and completed Flowers timing remain visible in that log.

## 10. Flowers-102 five-method sequential timing run (parallel alternative)

This performs two full-data epochs for each method in order
`KD -> CRD -> ReviewKD -> MGD -> OFA`. It measures all five methods without
committing to a full-run grouping.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Flowers-102 DeiT-Ti 5-method timing run` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_five_methods.py --dataset flowers102 --timing-run --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

The fixed Flowers teacher must print Top-1 `66.026996%` and SHA-256 prefix
`2b082bc98176...` during the precheck. Planned student training is 200 epochs
per method with batch size 64 and five warm-up epochs.

## 11. Chaoyang five-method sequential timing run (parallel alternative)

The mounted official dataset at `/app/data/chaoyang` is used automatically.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Chaoyang DeiT-Ti 5-method timing run` |
| 사용자 ID | `kau-aimslab` (연구실 계정) **or** `bapedragon` (개인 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_five_methods.py --dataset chaoyang --timing-run --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

The fixed Chaoyang teacher must print Top-1 `76.718093%` and SHA-256 prefix
`8cadc2d47131...`. Planned student training is 100 epochs per method with batch
size 64 and five warm-up epochs.

## 12. Choosing the efficient full-run packing

The two timing Issues are independent, so the fastest measurement is to run
Flowers with one account and Chaoyang with the other account at the same time.
Do not choose the final grouping before reading both returned timing summaries.

Each successful timing log ends with five individual estimates and one sum:

```text
[FINAL_RESULT] method=KD ... estimated_full=...
[FINAL_RESULT] method=CRD ... estimated_full=...
[FINAL_RESULT] method=ReviewKD ... estimated_full=...
[FINAL_RESULT] method=MGD ... estimated_full=...
[FINAL_RESULT] method=OFA ... estimated_full=...
[TIMING] estimated_selected_methods_full=...
[DONE] Selected methods completed successfully: KD,CRD,ReviewKD,MGD,OFA; resources may be released.
```

If the five-method sum is no more than about 540 minutes, all five may be run
in one full Issue. Otherwise split them with `--methods`, keeping every Issue
below about 540 minutes. This leaves roughly one hour under the 600-minute Pod
limit for setup, downloads, evaluation, checkpoint writes, and normal timing
variation. Example syntax (the actual grouping must follow the measured log):

```bash
python methods/run_five_methods.py --dataset flowers102 --full-run \
  --methods KD CRD ReviewKD --output-dir /app/output/flowers102_group1_v2 \
  --num-workers 4

python methods/run_five_methods.py --dataset flowers102 --full-run \
  --methods MGD OFA --output-dir /app/output/flowers102_group2_v2 \
  --num-workers 4
```

Every selected method has an independent result directory. If one method
fails, `five_method_status.json` records the error and all earlier completed
checkpoints and summaries remain under `/app/output`.

## 13. Combined full batch: Chaoyang five + Flowers five + CIFAR-100 KD

Build 449 measured the expected total as `7h 45m 59s`, leaving approximately
`2h 14m 01s` under the 600-minute Pod limit. The runner uses the short-first
order Chaoyang -> Flowers -> CIFAR-100 so ten shorter completed results are
already collected before the longest final task begins.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Flowers Chaoyang 5-method and CIFAR-100 KD full training` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_combined_full_batch.py --cifar-method KD --output-dir /app/output/combined_flowers_chaoyang_cifar100_kd_v2 --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

The output tree contains separate `chaoyang`, `flowers102`, and `cifar100_kd`
directories. Each method retains its own best/latest checkpoint and summary.
The root additionally contains `combined_batch_status.json` and
`combined_batch_summary.json`. If a later task fails, previously completed
directories are not deleted or overwritten.

## 14. Chaoyang Ours draft-matched paper-grid run

This request follows V3 rather than the supplied source's larger-grid rule.
The verified ResNet56 stages are `32 x 32`, `16 x 16`, and `8 x 8`; the DeiT
features must be bilinearly resampled to those three teacher grids.

### 14.1 Timing run (submit this first)

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Chaoyang DeiT-Ti Ours draft-matched timing run` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/Ours/chaoyang/train.py --timing-run --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Before launching the full run, require all of the following in the log:

```text
[MODE] ... timing_run=True ... planned_epochs=300
[PROTOCOL] name=chaoyang_deit_ti_ours_draftgrid_algbase_v3 ... warmup=20 ... batch=128 ...
[OURS] ... stage_grid=teacher ...
[BETA] schedule=alg_exact beta_on=2.5 ... threshold=-0.02 smoothing_window=50
[FEATURE_CHECK] ... stage_targets=[(2, 16, 32, 32), (2, 32, 16, 16), (2, 64, 8, 8)] ...
[DONE] Ours training completed successfully; resources may be released.
```

Also confirm the official Chaoyang split counts `4,021/2,139`, the fixed
teacher Top-1 `76.7181%` (approximately `76.72%`), and a passing teacher
runtime audit.

### 14.2 Full 300-epoch run

Submit only after the timing checklist passes.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Chaoyang DeiT-Ti Ours draft-matched full training` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/Ours/chaoyang/train.py --student-epochs 300 --batch-size 128 --warmup-epochs 20 --num-workers 4 --run-name ours_chaoyang_deit_ti_draftgrid_algbase_300ep_seed1 --output-dir /app/output` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

The comparison reference is `86.35%` Top-1. Report the raw measured best
Top-1; the code does not apply a teacher-gap correction. This request follows
the current draft's explicit uniform student schedule rather than the earlier
Chaoyang `100/64/5` dataset-specific profile.

## 15. Chaoyang ALG paper-matched run

This is the original ALG baseline, not Ours. It uses the public LG feature
matching path and ALG Eqs. (10)-(19). The paper comparison targets are
`83.50%` Top-1 and guidance stop epoch `108`.

### 15.1 Full-data two-epoch timing run (submit first)

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Chaoyang DeiT-Ti ALG researcher-sync timing run` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/ALG/chaoyang/train.py --timing-run --num-workers 4 --run-name alg_chaoyang_researcher_sync_timing_2ep --output-dir /app/output` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Require these markers before the full submission:

```text
[MODE] ... timing_run=True ... planned_epochs=300
[PROTOCOL] ... warmup=20 warmup_factor=0.001 ... batch=64 ...
[AUGMENT] ... auto_augment=rand-m9-mstd0.5-inc1 ...
[ALG] ... beta=2.5 tau=-0.02 smoothing_window=50 controller_warm_up=20 stop_condition=smoothed_derivative>tau descent_guard=False ...
[LG] ... student_blocks=(0,6,11) ... grid=larger_of_teacher_student ...
[FEATURE_CHECK] ... aligned=[(2, 16, 32, 32), (2, 32, 16, 16), (2, 64, 14, 14)] ...
[DONE] ALG training completed successfully; resources may be released.
```

Also verify the official split counts `4,021/2,139`, fixed teacher Top-1
approximately `76.72%`, and a passing native teacher audit.

### 15.2 Full 300-epoch run

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Chaoyang DeiT-Ti ALG researcher-sync 300-epoch training` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/ALG/chaoyang/train.py --output-dir /app/output --run-name alg_chaoyang_researcher_sync_300ep_seed1 --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

The full run saves independent `student_best.pt`, `student_latest.pt`, and
`summary.json` files. The final log prints measured Top-1 versus `83.50%` and
the observed guidance stop epoch versus `108`; no correction ratio is applied.

## 16. Chaoyang ALG on the historical draft-common base

This controlled run keeps the ALG operator and all draft-visible shared values
fixed while replacing the public LG/ALG augmentation/regularization base with
the historical common base used by the earlier Ours `81.11%` run. It is a
separate protocol family and must be compared only with that historical Ours
result.

### 16.1 Full-data two-epoch timing run

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Chaoyang DeiT-Ti ALG draft-common timing run` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/ALG/chaoyang/train_draft_common.py --timing-run --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Require these markers before the full run:

```text
[PROTOCOL] name=chaoyang_deit_ti_alg_draft_common_v1 ... base=draft_common ...
[ENV] ... amp=True seed=42 ...
[AUGMENT] RandomResizedCrop(scale=0.8..1.0)+HorizontalFlip ...
[FEATURE_CHECK] ... aligned=[(2, 16, 32, 32), (2, 32, 16, 16), (2, 64, 14, 14)] ...
[DONE] ALG training completed successfully; resources may be released.
```

### 16.2 Full 300-epoch run

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Chaoyang DeiT-Ti ALG draft-common full training` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/ALG/chaoyang/train_draft_common.py --output-dir /app/output --run-name alg_chaoyang_deit_ti_draft_common_300ep_seed42 --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

## 17. Researcher-sync Ours CIFAR-100

This entry matches the researcher-provided CIFAR-100 config: train/test batch
`64/200`, 300 epochs, AdamW `5e-4`, minimum LR `5e-6`, weight decay `0.05`,
20-epoch warm-up with factor `0.001`, drop path `0.1`, public LG strong
augmentation, FP32, teacher input 32, student input 224, and larger-grid
`32/16/14` matching. The researcher controller observes the complete Ours
guidance loss.

### 17.1 Timing run

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CIFAR-100 DeiT-Ti Ours researcher-sync timing run` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/Ours/cifar100/train.py --timing-run --num-workers 4 --run-name ours_cifar100_researcher_sync_timing_2ep --output-dir /app/output` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

### 17.2 Full 300-epoch run

Submit after the timing log confirms `batch=64`, `warmup=20`,
`min_lr=5e-06`, `drop_path=0.1`, `base=lg_official`, the `32/16/14` feature
targets, and an estimate below the 600-minute limit.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CIFAR-100 DeiT-Ti Ours researcher-sync 300-epoch training` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/Ours/cifar100/train.py --output-dir /app/output --run-name ours_cifar100_researcher_sync_300ep_seed1 --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

## 18. Flowers-102 + Chaoyang generic KD 300-epoch rerun

This request runs `KD -> CRD -> ReviewKD -> MGD -> OFA` for Chaoyang first,
then the same five for Flowers-102. It changes only the historical training
length (`100/200 -> 300`) and its cosine horizon. Batch 64, warm-up 5, seed 42,
augmentation, teacher hashes, adapters, and method losses remain fixed.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Flowers-102 Chaoyang generic KD five-method 300-epoch training` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_flowers_chaoyang_300ep.py --num-workers 4 --output-dir /app/output/generic_kd_flowers_chaoyang_300ep_seed42` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Measured estimate: Chaoyang `3h 41m 30s`, Flowers `4h 36m 48s`, total
`8h 18m 18s`. This leaves `1h 41m 42s` under the 600-minute limit. Every one
of the ten runs has its own directory, best/latest checkpoint, and summary;
earlier completed outputs remain intact if a later run fails.

The script installs its pinned `timm==1.0.27` dependency automatically, so no
extra installation command is required in the execution field.
