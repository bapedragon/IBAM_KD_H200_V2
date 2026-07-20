# IBAM KD H200 V2

Clean H200 experiment repository for CNN-to-ViT knowledge-distillation
experiments. The repository is being rebuilt from the teacher stage so that
LG, ALG, and Ours use the same low-resolution CNN guidance teacher.

## Current scope

The low-resolution teacher stage currently covers:

| Dataset | Teacher | Teacher input | Reference Top-1 |
|---|---|---:|---:|
| CIFAR-100 | CIFAR-style ResNet56 | **32 x 32** | 70.43% |
| Flowers-102 | CIFAR-style ResNet56 | **32 x 32** | 66.33% |
| Chaoyang | CIFAR-style ResNet56 | **32 x 32** | 77.20% |

The Flowers implementation uses the official `train+val` split (2,040 images)
for training and the official test split (6,149 images) for evaluation.

Student and KD method folders will be added only after the teacher checkpoint
has been validated. This keeps the new repository independent from the earlier
224 x 224 teacher experiments.

## Files

```text
IBAM_KD_H200_V2/
├── H200_ISSUE.md
├── README.md
├── PROTOCOL.md
├── requirements.txt
├── train_teacher_chaoyang.py
├── train_teacher_cifar100.py
└── train_teacher_flowers.py
```

The complete locked protocol and source audit are recorded in
[`PROTOCOL.md`](PROTOCOL.md).
Ready-to-copy H200 request values are recorded in
[`H200_ISSUE.md`](H200_ISSUE.md).

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
python train_teacher_cifar100.py --timing-run --num-workers 4
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
python train_teacher_cifar100.py --output-dir /app/output --run-name teacher_resnet56_cifar100_32_lg_official_seed1 --num-workers 4
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
python train_teacher_cifar100.py --smoke --num-workers 0
```

Smoke/timing accuracy is not a research result.

## Flowers-102 adjusted recipe v2

The first Flowers attempt inherited CIFAR's 300-epoch strong-augmentation
recipe and reached 59.33%, 7.00 points below the 66.33% reference. Because the
public LG commit does not contain its Flowers YAML, recipe v2 keeps the same
300-epoch schedule and changes only augmentation:

- the public LG weak-augmentation branch: resize to 32, random crop with
  padding 4, horizontal flip, and ImageNet normalization.

ResNet56, 32 x 32 input, scratch training, SGD 0.1, momentum 0.9, Nesterov,
weight decay 5e-4, batch size 128, cosine decay, and seed 1 remain unchanged.

Optional two-epoch timing check retaining the 300-epoch cosine schedule:

```bash
python train_teacher_flowers.py --timing-run --num-workers 4
```

For the collected full run, write to `/app/output`:

```bash
python train_teacher_flowers.py --output-dir /app/output --run-name teacher_resnet56_flowers102_32_weakaug_300ep_seed1 --num-workers 4
```

The full Flowers directory contains `best`, `latest`, and
`closest_to_reference` checkpoints plus `config.json`, `metrics.csv`, and
`summary.json`. The new run name prevents the 59.33% attempt from being
overwritten.

H200 build 439 verified the original Flowers data/model pipeline. Build 440
completed the first 300-epoch recipe at 59.33%; that checkpoint is retained as
an unsuccessful reproduction attempt. Timing-run accuracy is not a research
result.

## Chaoyang timing and full runs

Chaoyang is read from the persistent mount at `/app/data/chaoyang`. The script
validates all 4,021/2,139 JSON records, class counts, files, and 512 x 512 source
image format before training.

Run the full-data two-epoch timing check first:

```bash
python train_teacher_chaoyang.py --timing-run --num-workers 4
```

After it prints `[PROTOCOL_CHECK] status=PASS` and `[DONE]`, run:

```bash
python train_teacher_chaoyang.py --output-dir /app/output --run-name teacher_resnet56_chaoyang_32_moderateaug_300ep_seed1 --num-workers 4
```

The statistical recipe is fixed at ResNet56, 32 x 32, 300 epochs, SGD 0.1,
batch size 128, and seed 1. The exact Chaoyang teacher YAML is unavailable, so
the moderate crop policy is explicitly recorded as an implementation choice.

## Failure behavior

All important messages are printed with `flush=True`. Python exceptions print
a complete traceback and `[FATAL]`; successful completion prints `[DONE]`.
Checkpoints and summaries are rewritten atomically every completed epoch, so a
normal Python failure cannot leave a half-written checkpoint.
