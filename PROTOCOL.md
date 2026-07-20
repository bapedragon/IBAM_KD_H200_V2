# ResNet56 32 x 32 teacher protocols

This document records the exact protocol used by
`train_teacher_cifar100.py`. The goal is to reproduce the **32 x 32 guidance
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

## Flowers-102 locked protocol

`train_teacher_flowers.py` applies the same low-resolution guidance-teacher
recipe to Oxford Flowers-102. ALG's implementation details state that its
ResNet56 guidance CNN is trained on 32 x 32 images and uses SGD with learning
rate 0.1, momentum 0.9, weight decay 5e-4, and batch size 128. The LG dataset
code defines Flowers training as official `train+val` and evaluation as the
official test split.

| Item | Value |
|---|---:|
| Dataset | Oxford Flowers-102 |
| Train split | official train+val: 1,020+1,020=2,040 |
| Evaluation split | official test: 6,149 |
| Teacher | CIFAR-style ResNet56 (`6n+2`, `n=9`) |
| Input resolution | **32 x 32** |
| Number of classes | 102 |
| Epochs | 300 |
| Train / test batch size | 128 / 200 |
| Optimizer | SGD |
| Initial learning rate | 0.1 |
| Momentum / Nesterov | 0.9 / enabled |
| Weight decay | 5e-4, excluding bias and normalization parameters |
| LR schedule | cosine decay to 0; no warm-up |
| Label smoothing / Mixup / CutMix | 0 / disabled / disabled |
| Mixed precision | disabled |
| Seed / cuDNN benchmark | 1 / disabled |
| Primary metric | official test Top-1 accuracy |
| Reference teacher Top-1 | 66.33% |

Training and evaluation transforms are the same official LG transforms
recorded above, with the spatial size set to 32 x 32. The Flowers-specific
public YAML is not present at the audited LG commit; consequently, the shared
LG code behavior and ALG's explicit implementation details are the documented
reproduction boundary. All listed statistical values are constants rather
than command-line overrides.

## H200 timing verification

| Dataset | H200 build | Account | Full-data epochs | Average epoch | Estimated full run | Result |
|---|---:|---|---:|---:|---:|---|
| CIFAR-100 | 437 | `bapedragon` | 2 | 9.3 s | 46m 17s | PASS |
| Flowers-102 | 439 | `kau-aimslab` | 2 | 5.3 s | 26m 25s | PASS |

Both timing runs retained their locked 300-epoch cosine schedule. Their
2-epoch accuracies are startup diagnostics and are not reported as research
results. Build 439 additionally verified the official Flowers downloads and
MD5 values, the 2,040/6,149 split, 861,750 parameters, 32/16/8 feature grids,
and atomic creation of all checkpoint variants and summary files.
