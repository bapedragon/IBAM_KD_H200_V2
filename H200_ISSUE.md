# H200 Issue request values

Use the timing request first when a pipeline is new. CIFAR-100 estimates its
300-epoch duration. The Flowers pipeline has already completed multiple full
runs, so its final 450-epoch request may be submitted directly.

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

## CIFAR-100 Ours + CRD + MGD sequential timing run

Ours has not yet been measured with the fixed V2 teacher, so submit this
timing request before deciding whether all three full runs fit safely inside
the 600-minute Pod limit.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CIFAR-100 DeiT-Ti Ours CRD MGD timing run` |
| 사용자 ID | `bapedragon` (개인 계정) **or** `kau-aimslab` (연구실 계정) |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python methods/run_cifar100_ours_crd_mgd.py --timing-run --output-dir /app/output/cifar100_ours_crd_mgd_timing_v2 --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Success requires the individual Ours, CRD, and MGD `[DONE]` messages followed
by `[DONE] Ours, CRD, and MGD completed successfully`. Do not submit the
corresponding `--full-run` until this log reports the three-method aggregate
estimate.

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
