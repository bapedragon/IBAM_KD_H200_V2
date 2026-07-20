# H200 Issue request values

Use the timing request first when a pipeline is new. CIFAR-100 estimates its
300-epoch duration; the adjusted Flowers recipe estimates its 200-epoch
duration without placing temporary artifacts in the collected `/app/output`
directory.

## 1. Timing run

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 CIFAR-100 ResNet56 32x32 teacher timing run` |
| 사용자 ID | `bapedragon` |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python train_teacher_cifar100.py --timing-run --num-workers 4` |
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
| 코드 실행 명령어 | `python train_teacher_cifar100.py --output-dir /app/output --run-name teacher_resnet56_cifar100_32_lg_official_seed1 --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

Do not change `/app/output` to `/app/outputs`; the H200 collector uses the
singular path.

## 3. Flowers-102 adjusted-recipe timing run (`kau-aimslab`, optional)

This request can run in parallel with the CIFAR-100 job because it uses a
different GitHub user allocation and writes only temporary timing artifacts.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Flowers-102 ResNet56 32x32 adjusted teacher timing run` |
| 사용자 ID | `kau-aimslab` |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python train_teacher_flowers.py --timing-run --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |

## 4. Flowers-102 adjusted full 200-epoch run (`kau-aimslab`)

The previous Flowers run already verified the H200 data/model pipeline. The
new timing request is therefore optional; the full request may be submitted
directly after the updated commit is visible on GitHub.

| Field | Value |
|---|---|
| Title | `[Request]: 박철현 Flowers-102 ResNet56 32x32 adjusted teacher training` |
| 사용자 ID | `kau-aimslab` |
| 실행할 코드의 GitHub 링크 | `https://github.com/bapedragon/IBAM_KD_H200_V2.git` |
| 코드 실행 명령어 | `python train_teacher_flowers.py --output-dir /app/output --run-name teacher_resnet56_flowers102_32_weakaug_200ep_seed1 --num-workers 4` |
| 사용할 이미지 | `pytorch/pytorch:latest` |
| 사용 언어 | `Python` |
| GPU 할당량 (MIG 개수) | `7` |
