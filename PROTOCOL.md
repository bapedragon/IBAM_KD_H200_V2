# ResNet56 32 x 32 teacher protocols

This document records the exact protocol used by
`teachers/train_teacher_cifar100.py`. The goal is to reproduce the **32 x 32 guidance
teacher** used by LG and ALG, not the previously trained 224 x 224 teacher.

## Paper and official-code evidence

- The LG paper explicitly describes ResNet56 as the guidance model trained at
  32 x 32 resolution. Its Table 1 reports `ResNet-56 (32 res.)` with 70.43%
  Top-1 on CIFAR-100.
- The official LG configuration is
  `configs/resnet/r-56_c100.yaml` in
  <https://github.com/lkhl/tiny-transformers>.
- Audited official repository commit:
  `d2165f74049c906b0afc9f957491960fb3c0cc8b`.

## Locked experiment settings

| Item | Value |
|---|---:|
| Dataset | CIFAR-100 train 50,000 / test 10,000 |
| Teacher | CIFAR-style ResNet56 (`6n+2`, `n=9`) |
| Input resolution | **32 x 32** |
| Number of classes | 100 |
| Epochs | 300 |
| Train batch size | 128 |
| Test batch size | 200 |
| Optimizer | SGD |
| Initial learning rate | 0.1 |
| Momentum | 0.9 |
| Nesterov | enabled |
| Weight decay | 5e-4, excluding bias and normalization parameters |
| LR schedule | cosine decay to 0 |
| Warm-up | none |
| Label smoothing | 0 |
| Mixup / CutMix | disabled |
| Mixed precision | disabled by default |
| Seed | 1 |
| cuDNN benchmark | disabled |
| Primary metric | CIFAR-100 test Top-1 accuracy |
| LG reference Top-1 | 70.43% |

These statistical protocol values are constants in the training code and are
not exposed as command-line overrides. Only operational values such as paths,
run name, worker count, and smoke/timing mode can be changed.

## Official strong augmentation

The official configuration inherits `TRAIN.STRONG_AUGMENTATION=True`. This
implementation calls `timm.create_transform` with the same arguments as the
official LG code:

- random resized crop to 32 x 32 with bicubic interpolation;
- random horizontal flip;
- RandAugment `rand-m9-mstd0.5-inc1`;
- ImageNet mean/std normalization;
- random erasing probability 0.25, pixel mode, one region.

The official call passes `color_jitter=0.4`; with RandAugment enabled,
`timm==1.0.27` realizes the secondary transform as RandAugment without a
separate ColorJitter operation. This realized behavior is intentionally left
unchanged and is visible in the startup audit log.

Evaluation resizes directly to 32 x 32 and uses ImageNet mean/std
normalization, matching the official LG dataset code.

## Operational additions

The following do not change the model objective or training recipe:

- verified CIFAR-100 download with mirror fallback and official MD5 checks;
- H200/Python/PyTorch environment logging;
- per-epoch timing and metrics output;
- atomic `best`, `latest`, and `closest-to-reference` checkpoint writes;
- a JSON summary, CSV history, and SHA-256 hashes.

`closest-to-reference` is saved only as a diagnostic convenience. The primary
experimental result remains the highest test Top-1 checkpoint (`best`).

## Known reproducibility boundary

The original LG environment used older PyTorch/CUDA packages, while the H200
runner uses a modern PyTorch image and installs `timm==1.0.27`. The architecture,
configuration values, and augmentation arguments are matched, but small
numerical differences across library and GPU versions are still possible.

## Pre-run verification completed

Before publication to this repository, the standalone implementation was
instantiated beside the official LG ResNet56 implementation. Verification
confirmed:

- identical parameter count: 861,620;
- identical ordered state-dict keys: 344/344;
- identical tensor shapes for every state-dict entry;
- feature shapes `32x32`, `16x16`, and `8x8` for stages 1-3;
- PyTorch cosine scheduler values equal to the official timm cosine scheduler
  at checked epochs, including epochs 1, 2, 20, 100, 299, and 300;
- local smoke run completed forward, backward, evaluation, atomic checkpoint,
  metrics, summary, and SHA-256 generation.

## Flowers-102 reproduction boundary and final recipe v5

ALG's implementation details state that its ResNet56 guidance CNN is trained
on 32 x 32 images and uses SGD with learning rate 0.1, momentum 0.9, weight
decay 5e-4, and batch size 128. The LG dataset code defines Flowers training
as official `train+val` and evaluation as the official test split.

The audited public LG commit contains no Flowers YAML. In particular, the
paper's 300-epoch statement describes the 224 x 224 ViT students, not the
Flowers ResNet56 teacher. Therefore the exact Flowers teacher epoch count and
strong-augmentation switch cannot be claimed as published settings.

The first H200 attempt used the public LG strong-augmentation path for 300
epochs and reached 59.33% Top-1. Its training accuracy was only 66.04% at
epoch 300 and its test accuracy was still improving (52.85% at epoch 200,
57.59% at epoch 250, and 59.33% at epoch 299). The weak-augmentation control
then reached only 51.63% despite 100% training accuracy, demonstrating severe
overfitting. It is rejected.

The 600-epoch strong run subsequently reached a best Top-1 of 69.87% and a
closest-to-draft checkpoint of 66.35% at epoch 457. An independent 400-epoch
strong run reached 63.33% at epoch 346. Recipe v5 keeps every statistical
choice unchanged except that it uses an independent 450-epoch cosine schedule.
It is the final explicit implementation comparison among the completed 300-,
400-, and 600-epoch schedules, not an official LG Flowers setting.
The primary target from the current draft is 66.33%. For provenance, the
original LG paper's Table 1 reports 59.83% for its 32-resolution Flowers
ResNet56 teacher. These reference values are logged separately.

| Item | Value |
|---|---:|
| Dataset | Oxford Flowers-102 |
| Train split | official train+val: 1,020+1,020=2,040 |
| Evaluation split | official test: 6,149 |
| Teacher | CIFAR-style ResNet56 (`6n+2`, `n=9`) |
| Input resolution | **32 x 32** |
| Number of classes | 102 |
| Epochs | **450** |
| Train / test batch size | 128 / 200 |
| Optimizer | SGD |
| Initial learning rate | 0.1 |
| Momentum / Nesterov | 0.9 / enabled |
| Weight decay | 5e-4, excluding bias and normalization parameters |
| LR schedule | cosine decay to 0; no warm-up |
| Label smoothing / Mixup / CutMix | 0 / disabled / disabled |
| Mixed precision | disabled |
| Seed / cuDNN benchmark | 1 / disabled |
| Train augmentation | official LG strong path: random resized crop 32, bicubic, horizontal flip, RandAugment m9, random erasing p=0.25 |
| Primary metric | official test Top-1 accuracy |
| Draft reference teacher Top-1 | 66.33% |
| Published LG teacher Top-1 | 59.83% |

Evaluation remains direct resize to 32 x 32 plus ImageNet normalization. All
listed statistical values are constants rather than command-line overrides.
The 450-epoch schedule must be described as an implementation choice rather
than an exact official Flowers reproduction.

## H200 timing verification

| Dataset | H200 build | Account | Full-data epochs | Average epoch | Estimated full run | Result |
|---|---:|---|---:|---:|---:|---|
| CIFAR-100 | 437 | `bapedragon` | 2 | 9.3 s | 46m 17s | PASS |
| Flowers-102 | 439 | `kau-aimslab` | 2 | 5.3 s | 26m 25s | PASS |
| Chaoyang | 442 | `bapedragon` | 2 | 5.2 s | 26m 11s | PASS |

The timing runs used the original 300-epoch attempt-1 schedule. Their 2-epoch
accuracies are startup diagnostics and are not research results. Build 439
verified the official Flowers downloads and MD5 values, the 2,040/6,149 split,
861,750 parameters, 32/16/8 feature grids, and atomic checkpoint creation.

## Completed full-run results

| Dataset / recipe | H200 build | Epochs | Best Top-1 | Reference | Gap | Status |
|---|---:|---:|---:|---:|---:|---|
| CIFAR-100 official recipe | 438 | 300 | 71.91% | 70.43% | +1.48 pp | accepted |
| CIFAR-100 closest diagnostic | 438 | - | 70.53% | 70.43% | +0.10 pp | saved |
| Flowers attempt 1, strong augmentation | 440 | 300 | 59.33% | draft 66.33% | -7.00 pp | retained baseline |
| Flowers attempt 1 vs published LG | 440 | 300 | 59.33% | published 59.83% | -0.50 pp | comparable |
| Flowers attempt 2, weak augmentation | 441 | 300 | 51.63% | draft 66.33% | -14.70 pp | rejected: overfit |
| Flowers attempt 3, strong augmentation | 444 | 600 | 69.87% | draft 66.33% | +3.54 pp | completed |
| Flowers attempt 3, closest diagnostic | 444 | epoch 457 | 66.35% | draft 66.33% | +0.02 pp | saved |
| Flowers attempt 4, strong augmentation | - | 400 | 63.33% | draft 66.33% | -3.00 pp | completed |
| Flowers final, strong augmentation | 447 | 450 | 66.03% | draft 66.33% | -0.30 pp | **selected best** |
| Chaoyang moderate augmentation | 443 | 300 | 76.72% | 77.20% | -0.48 pp | **selected best** |

The selected checkpoints are recorded in `teachers/checkpoints/manifest.json`.
All downstream KD methods must reuse these exact hashes rather than selecting
a different teacher per method.

## CIFAR-100 DeiT-Ti common student protocol (V2)

KD, CRD, and ReviewKD are compared under one fixed base protocol. Only the
method-specific transfer loss and its documented adapter may differ.

| Item | Value |
|---|---:|
| Student | DeiT-Ti (`deit_tiny_patch16_224`) |
| Initialization | scratch; no external pretrained weights |
| Student input | **224 x 224** |
| Teacher input | **32 x 32** |
| Epochs | 300 |
| Batch size | 128 |
| Optimizer | AdamW |
| Initial LR | `5e-4` |
| Weight decay | `0.05` |
| LR schedule | 20-epoch warm-up, then cosine decay |
| Label smoothing | `0.1` |
| AMP | enabled on CUDA |
| Seed | 42 |
| Metric | CIFAR-100 test Top-1 |

Training uses random resized crop to 224 (`scale=0.8-1.0`, bicubic) and random
horizontal flip. Evaluation uses resize to 256 and center crop to 224. The
student tensor uses CIFAR-100 normalization. For the teacher branch, that same
tensor is converted back to image space, bilinearly resized to 32, and then
normalized with the ImageNet statistics used during teacher training. This
preserves identical crop/flip geometry while respecting each model's training
normalization.

The exact KD coefficients, CRD official settings, and ReviewKD feature adapter
are separately recorded in `methods/KD/README.md`, `methods/CRD/README.md`, and
`methods/ReviewKD/README.md`. These README files are part of the locked
experimental record.

## Chaoyang locked protocol

ALG explicitly identifies the guidance teacher as ResNet56 trained from
scratch at 32 x 32, with SGD learning rate 0.1, momentum 0.9, weight decay
5e-4, and batch size 128. The draft reference teacher Top-1 is 77.20%.

The audited LG commit does not contain a Chaoyang teacher YAML. Consequently,
the 300-epoch schedule and augmentation below are documented implementation
choices. The crop policy is the moderate policy already validated on the
mounted Chaoyang images, scaled from 224 to the required 32 x 32 input.

| Item | Value |
|---|---:|
| Dataset | Chaoyang |
| Train / test split | official 4,021 / 2,139 |
| Classes | normal, serrated, adenocarcinoma, adenoma |
| Teacher | CIFAR-style ResNet56 (`6n+2`, `n=9`) |
| Input resolution | **32 x 32** |
| Epochs | **300** |
| Train / test batch size | 128 / 200 |
| Optimizer | SGD |
| Initial learning rate | 0.1 |
| Momentum / Nesterov | 0.9 / enabled |
| Weight decay | 5e-4, excluding bias and normalization parameters |
| LR schedule | cosine decay to 0; no warm-up |
| Mixed precision | disabled |
| Seed / cuDNN benchmark | 1 / disabled |
| Train augmentation | random resized crop 32, scale 0.8-1.0, bicubic; horizontal flip |
| Evaluation | direct resize to 32, ImageNet normalization |
| Reference teacher Top-1 | 77.20% |

The H200 script resolves one nested extraction directory when present and
requires the exact official per-class counts before any optimizer step. It
writes atomic best/latest/closest checkpoints, metrics, config, summary, and
SHA-256 hashes.
