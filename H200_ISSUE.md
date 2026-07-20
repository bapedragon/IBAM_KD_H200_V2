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
