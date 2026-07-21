# V2 locked teacher and student protocols

This document records the selected 32 x 32 ResNet56 teacher recipes for all
three datasets and the shared 224 x 224 DeiT-Ti student protocols used by every
compared method.

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

## Flowers-102 selected 450-epoch teacher protocol

ALG's implementation details state that its ResNet56 guidance CNN is trained
on 32 x 32 images and uses SGD with learning rate 0.1, momentum 0.9, weight
decay 5e-4, and batch size 128. The LG dataset code defines Flowers training
as official `train+val` and evaluation as the official test split.

The audited public LG commit contains no Flowers YAML. In particular, the
paper's 300-epoch statement describes the 224 x 224 ViT students, not the
Flowers ResNet56 teacher. Therefore the exact Flowers teacher epoch count and
strong-augmentation switch cannot be claimed as published settings.

The selected implementation uses the public LG strong-augmentation path and
an independent 450-epoch cosine schedule. The primary target from the current
draft is 66.33%. The selected best checkpoint reaches 66.03% at epoch 389.
For provenance, the original LG paper's Table 1 reports 59.83% for its
32-resolution Flowers ResNet56 teacher. The 450-epoch count remains an explicit
implementation choice rather than a published LG Flowers setting.

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

## Selected full-run teacher results

| Dataset / recipe | Epochs | Average epoch | Full-run elapsed | Best Top-1 | Reference | Gap | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| CIFAR-100 official recipe | 300 | 8.75 s | 44m 01s | 71.91% | 70.43% | +1.48 pp | selected |
| Flowers-102 strong augmentation | 450 | 5.02 s | 38m 11s | 66.03% | 66.33% | -0.30 pp | selected |
| Chaoyang moderate augmentation | 300 | 4.87 s | 24m 34s | 76.72% | 77.20% | -0.48 pp | selected |

The selected checkpoints are recorded in `teachers/checkpoints/manifest.json`.
All downstream KD methods must reuse these exact hashes rather than selecting
a different teacher per method.

## DeiT-Ti common student protocols (V2)

KD, CRD, ReviewKD, MGD, and OFA are compared under one fixed base protocol per
dataset. Only the method-specific transfer loss and its documented adapter may
differ.

| Item | Value |
|---|---:|
| Student | DeiT-Ti (`deit_tiny_patch16_224`) |
| Initialization | scratch; no external pretrained weights |
| Student input | **224 x 224** |
| Teacher input | **32 x 32** |
| Epochs | CIFAR-100 300 / Flowers-102 200 / Chaoyang 100 |
| Batch size | CIFAR-100 128 / Flowers-102 64 / Chaoyang 64 |
| Optimizer | AdamW |
| Initial LR | `5e-4` |
| Weight decay | `0.05` |
| LR schedule | cosine decay; warm-up 20 (CIFAR-100) or 5 (Flowers/Chaoyang) |
| Label smoothing | `0.1` |
| AMP | enabled on CUDA |
| Seed | 42 |
| Metric | official dataset test Top-1 |

Training uses random resized crop to 224 (`scale=0.8-1.0`, bicubic) and random
horizontal flip. Evaluation uses resize to 256 and center crop to 224. The
student tensor uses CIFAR-100 normalization on CIFAR-100 and ImageNet
normalization on Flowers/Chaoyang. For the teacher branch, that same tensor is
converted back to image space, bilinearly resized to 32, and then normalized
with the ImageNet statistics used during teacher training. This preserves
identical crop/flip geometry while respecting each model's normalization.

The exact method coefficients, official-code sources, and heterogeneous
adapters are separately recorded under `methods/KD`, `methods/CRD`,
`methods/ReviewKD`, `methods/MGD`, and `methods/OFA`. These README files are
part of the locked experimental record.

### Generic-KD 300-epoch comparison override

The new comparison run uses 300 epochs for Flowers-102 and Chaoyang as well as
CIFAR-100. This does not adopt the Ours/ALG data or regularization config.
Instead, it deliberately preserves each generic-KD dataset protocol above and
changes only:

1. the optimizer-step training length to 300 epochs; and
2. the cosine scheduler horizon to the same 300 epochs.

Flowers and Chaoyang therefore retain batch 64, warm-up 5, label smoothing
0.1, CUDA AMP, seed 42, the same transforms and fixed teacher hash. KD, CRD,
ReviewKD, MGD, and OFA also retain their existing transfer operators,
coefficients, and CNN-to-ViT adapters. This controlled override is identified
in saved configs as `*_300ep_epoch_only_v1`.

Measured H200 averages predict `3h 41m 30s` for all five Chaoyang runs and
`4h 36m 48s` for all five Flowers runs, totaling `8h 18m 18s`.

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

## Chaoyang Adaptive Locality Guidance (ALG)

ALG is recorded separately from both generic KD and Ours. It retains the
public LG feature objective and uses the adaptive stop rule in ALG
Eqs. (10)-(19).

| Item | Value | Source |
|---|---:|---|
| Student / initialization | DeiT-Ti / scratch | ALG paper, LG config |
| Student / teacher input | 224 / 32 | ALG paper, LG config |
| Epochs / train batch | 300 / 128 | ALG paper, LG config |
| Optimizer | AdamW | ALG paper |
| LR / minimum LR | `5e-4` / `5e-6` | ALG paper, LG config |
| Weight decay | `0.05` | ALG paper |
| Warm-up | 20 epochs, factor `0.001` | ALG paper, LG config |
| Drop path | `0.1` | LG DeiT-Ti config |
| Label smoothing / Mixup / CutMix | `0` / off / off | LG defaults |
| FP mode / seed | FP32 / `1` | LG defaults |
| Teacher stages | `[0,1,2]` | LG config |
| Student blocks | `[0,6,11]` | LG config |
| Feature operator | 1x1 projection, larger-grid bilinear resize, summed MSE | LG code |
| Beta / threshold / windows | `2.5` / `-0.02` / `50,50` | ALG paper |
| Paper target / stop epoch | `83.50%` / `108` | ALG Table II |

Strong augmentation is the public LG path: color-jitter argument `0.4`,
RandAugment `rand-m9-mstd0.5-inc1`, random erasing `0.25` pixel mode,
bicubic interpolation, and ImageNet normalization. Evaluation directly
resizes the complete image to 224 x 224.

The remaining reproduction boundaries are the newer timm/PyTorch environment,
the local fixed teacher (`76.72%` versus the original LG teacher's `78.12%`),
the epoch-1 derivative initialization, and best-checkpoint reporting. They are
saved in every ALG checkpoint and summary.
