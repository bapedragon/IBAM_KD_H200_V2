# IBAM KD H200 V2

Clean H200 experiment repository for CNN-to-ViT knowledge-distillation
experiments. The repository is being rebuilt from the teacher stage so that
LG, ALG, and Ours use the same low-resolution CNN guidance teacher.

## Current scope

The low-resolution teacher stage currently covers:

| Dataset | Teacher input | Selected Top-1 | Reference | Gap |
|---|---:|---:|---:|---:|
| CIFAR-100 | **32 x 32** | **71.91%** | 70.43% | +1.48 pp |
| Flowers-102 | **32 x 32** | **66.03%** | 66.33% | -0.30 pp |
| Chaoyang | **32 x 32** | **76.72%** | 77.20% | -0.48 pp |

The Flowers implementation uses the official `train+val` split (2,040 images)
for training and the official test split (6,149 images) for evaluation.

All three primary `best` checkpoints have passed SHA-256, strict state-dict,
metadata, and 32 x 32 forward checks. They are fixed before downstream KD and
must be reused across every compared method. This repository remains
independent from the earlier 224 x 224 teacher experiments.

## Files

```text
IBAM_KD_H200_V2/
├── H200_ISSUE.md
├── README.md
├── PROTOCOL.md
├── requirements.txt
├── methods/
│   ├── README.md
│   ├── run_cifar100_three_methods.py
│   ├── run_combined_full_batch.py
│   ├── run_five_methods.py
│   ├── run_flowers_chaoyang_timing.py
│   ├── KD/
│   ├── CRD/
│   ├── ReviewKD/
│   ├── MGD/
│   └── OFA/
└── teachers/
    ├── checkpoints/
    │   ├── cifar100/
    │   ├── flowers102/
    │   ├── chaoyang/
    │   ├── README.md
    │   └── manifest.json
    ├── README.md
    ├── train_teacher_chaoyang.py
    ├── train_teacher_cifar100.py
    ├── train_teacher_flowers.py
    └── verify_checkpoints.py
```

The complete locked protocol and source audit are recorded in
[`PROTOCOL.md`](PROTOCOL.md).
Ready-to-copy H200 request values are recorded in
[`H200_ISSUE.md`](H200_ISSUE.md).

## DeiT-Ti student stage

The V2 student pipeline supports all five generic methods in the draft table:
`KD -> CRD -> ReviewKD -> MGD -> OFA`. Every method reuses the selected fixed
teacher hash for its dataset while training a scratch DeiT-Ti at 224 x 224.

The teacher input is derived from the same augmented student tensor using
bilinear resize to 32 x 32, so crop and flip geometry cannot drift between the
two branches. The teacher and student normalizations are applied separately.
Spatial feature methods bilinearly match the CNN feature grid to the DeiT
14x14 patch grid where required.

Run the full-data two-epoch timing sequence first. Flowers and Chaoyang can be
submitted as separate Issues in parallel using the personal and lab accounts:

```bash
python methods/run_five_methods.py --dataset flowers102 --timing-run \
  --output-dir /app/output/flowers102_five_methods_timing_v2 --num-workers 4

python methods/run_five_methods.py --dataset chaoyang --timing-run \
  --output-dir /app/output/chaoyang_five_methods_timing_v2 --num-workers 4
```

Alternatively, one Issue can measure all ten dataset-method combinations:

```bash
python methods/run_flowers_chaoyang_timing.py \
  --num-workers 4
```

The timing artifacts stay in the temporary clone by default; all duration
estimates needed for job packing are printed in the Issue log. Full training
must instead use an explicit singular `/app/output/...` collection path.

The measured Flowers and Chaoyang total is 4h 39m 31s. One measured CIFAR-100
method can therefore be appended safely in the same 600-minute Pod. The locked
short-first full batch runs Chaoyang five methods, Flowers five methods, and
then CIFAR-100 KD:

```bash
python methods/run_combined_full_batch.py --cifar-method KD \
  --output-dir /app/output/combined_flowers_chaoyang_cifar100_kd_v2 \
  --num-workers 4
```

The expected total is 7h 45m 59s, leaving approximately 2h 14m under the Pod
limit. Dataset and method directories remain independent throughout the batch.

Each method writes its own `student_best.pt`, `student_latest.pt`, and
`summary.json` under a distinct run directory. The runner additionally writes
`five_method_status.json` and `five_method_summary.json`; therefore no method
can overwrite another method's files. Its timing summary provides individual
and combined full-run estimates for packing jobs safely below the 600-minute
Pod limit. See [`methods/README.md`](methods/README.md) for the locked base
protocols and each method directory for exact losses, official-code provenance,
and heterogeneous adapters.

## Fixed teachers for downstream KD

The selected weights and their full provenance are under
[`teachers/checkpoints`](teachers/checkpoints). Before launching a KD job, run:

```bash
python teachers/verify_checkpoints.py --dataset all
```

This verifies each committed SHA-256, checkpoint metadata, strict model load,
output dimensions, and finite 32 x 32 inference. The loader also freezes the
returned teacher parameters for downstream use.

## Why 32 x 32?

The original LG paper explicitly uses a ResNet56 guidance model trained on
32 x 32 images. The ViT student is trained separately at 224 x 224. These are
two different input branches and should not be confused with the 224 x 224
ResNet18 CNN baseline in the LG paper.

Sources:

- LG paper: <https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840108.pdf>
- LG official code: <https://github.com/lkhl/tiny-transformers>
- Audited official commit: `d2165f74049c906b0afc9f957491960fb3c0cc8b`

## H200 timing run (recommended first)

The timing run uses the full CIFAR-100 dataset for two epochs while keeping the
300-epoch cosine schedule. Its artifacts remain inside the temporary cloned
repository and therefore do not need to be collected.

```bash
python teachers/train_teacher_cifar100.py --timing-run --num-workers 4
```

Expected log markers:

- `[PROTOCOL_CHECK] status=PASS`
- `[MODEL] teacher_params=861,620`
- `[DATA] train_samples=50000 test_samples=10000`
- `[TIMING] estimated_300_teacher=...`
- `[DONE] Teacher training completed successfully`

## H200 full training

After the timing run succeeds, use:

```bash
python teachers/train_teacher_cifar100.py --output-dir /app/output --run-name teacher_resnet56_cifar100_32_lg_official_seed1 --num-workers 4
```

The core full-run values are already fixed to the official settings, so the
command does not need to repeat epoch, batch-size, image-size, learning-rate,
or seed arguments.

## Collected artifacts

The full command writes the following directory:

```text
/app/output/teacher_resnet56_cifar100_32_lg_official_seed1/
├── teacher_resnet56_cifar100_32_best.pt
├── teacher_resnet56_cifar100_32_latest.pt
├── teacher_resnet56_cifar100_32_closest_to_lg_reference.pt
├── config.json
├── metrics.csv
└── summary.json
```

- `best.pt`: highest test Top-1; primary teacher checkpoint.
- `latest.pt`: epoch 300 state.
- `closest_to_lg_reference.pt`: closest observed result to 70.43%; diagnostic
  only, not the primary reported result.
- `summary.json`: final accuracy, timing, paths, hashes, and protocol metadata.

Checkpoints contain both `model_state` (official LG-style key) and `model`
aliases, as well as accuracy, epoch, optimizer state, architecture metadata,
and preprocessing metadata for downstream loading.

## Local smoke test

This checks imports, model forward/backward, data preparation, and checkpoint
creation on deterministic subsets:

```bash
python teachers/train_teacher_cifar100.py --smoke --num-workers 0
```

Smoke/timing accuracy is not a research result.

## Flowers-102 final recipe v5

The 300-epoch official-strong run reached 59.33%. A weak-augmentation control
reached only 51.63% while fitting the training set to 100%, so it is rejected.
The 600-epoch strong run reached 69.87%, with a 66.35% draft-matching
checkpoint at epoch 457. The independent 400-epoch comparison then reached
63.33% at epoch 346. The final 450-epoch comparison keeps the public LG strong
augmentation and starts a new cosine schedule from scratch:

- random resized crop to 32 with bicubic interpolation;
- horizontal flip;
- RandAugment `rand-m9-mstd0.5-inc1`;
- random erasing probability 0.25;
- ImageNet normalization.

ResNet56, 32 x 32 input, scratch training, SGD 0.1, momentum 0.9, Nesterov,
weight decay 5e-4, batch size 128, cosine decay, and seed 1 remain unchanged.
The draft target (66.33%) and the original LG paper value (59.83%) are logged
separately. The 450-epoch schedule is a documented implementation choice
because the public LG repository has no Flowers teacher YAML.

Optional two-epoch timing check retaining the 450-epoch cosine schedule:

```bash
python teachers/train_teacher_flowers.py --timing-run --num-workers 4
```

For the collected full run, write to `/app/output`:

```bash
python teachers/train_teacher_flowers.py --output-dir /app/output --run-name teacher_resnet56_flowers102_32_strongaug_450ep_seed1 --num-workers 4
```

The full Flowers directory contains `best`, `latest`, and
`closest_to_reference` checkpoints plus `config.json`, `metrics.csv`, and
`summary.json`. The new run name prevents all prior attempts from being
overwritten.

H200 build 439 verified the original Flowers data/model pipeline. Build 440
completed the strong 300-epoch recipe at 59.33%; build 441 completed the weak
control at 51.63%; build 444 completed the strong 600-epoch run at 69.87%.
The independent 400-epoch run completed at 63.33%. The 450-epoch run is the
last scheduled Flowers teacher comparison.
Timing-run accuracy is not a research result.

## Chaoyang timing and full runs

Chaoyang is read from the persistent mount at `/app/data/chaoyang`. The script
validates all 4,021/2,139 JSON records, class counts, files, and 512 x 512 source
image format before training.

Run the full-data two-epoch timing check first:

```bash
python teachers/train_teacher_chaoyang.py --timing-run --num-workers 4
```

After it prints `[PROTOCOL_CHECK] status=PASS` and `[DONE]`, run:

```bash
python teachers/train_teacher_chaoyang.py --output-dir /app/output --run-name teacher_resnet56_chaoyang_32_moderateaug_300ep_seed1 --num-workers 4
```

The statistical recipe is fixed at ResNet56, 32 x 32, 300 epochs, SGD 0.1,
batch size 128, and seed 1. The exact Chaoyang teacher YAML is unavailable, so
the moderate crop policy is explicitly recorded as an implementation choice.

## Failure behavior

All important messages are printed with `flush=True`. Python exceptions print
a complete traceback and `[FATAL]`; successful completion prints `[DONE]`.
Checkpoints and summaries are rewritten atomically every completed epoch, so a
normal Python failure cannot leave a half-written checkpoint.
