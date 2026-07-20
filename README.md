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

## Flowers-102 timing and full runs

Run two full-dataset timing epochs while retaining the locked 300-epoch cosine
schedule:

```bash
python train_teacher_flowers.py --timing-run --num-workers 4
```

After the timing log passes, collect a full run under `/app/output`:

```bash
python train_teacher_flowers.py --output-dir /app/output --run-name teacher_resnet56_flowers102_32_lg_official_seed1 --num-workers 4
```

The full Flowers directory contains `best`, `latest`, and
`closest_to_reference` checkpoints plus `config.json`, `metrics.csv`, and
`summary.json`. Core statistical settings are locked in code exactly as for
the CIFAR-100 teacher.

## Failure behavior

All important messages are printed with `flush=True`. Python exceptions print
a complete traceback and `[FATAL]`; successful completion prints `[DONE]`.
Checkpoints and summaries are rewritten atomically every completed epoch, so a
normal Python failure cannot leave a half-written checkpoint.
