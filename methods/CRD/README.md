# Contrastive Representation Distillation (CRD)

The CRD criterion is ported from the authors' official RepDistiller code.

- Official repository: https://github.com/HobbitLong/RepDistiller
- Pinned commit: `b84f547c5db6a35318d4671d7d5c4de74c822403`
- License: BSD 2-Clause (`OFFICIAL_CODE_LICENSE.txt`)

## Fixed method settings

| Setting | Value |
|---|---:|
| CE weight | `1.0` |
| Logit-KD weight | `0.0` |
| CRD weight | `0.8` |
| Projection dimension | 128 |
| Negative samples | 16,384 |
| NCE temperature | `0.07` |
| Memory momentum | `0.5` |
| Sampling mode | exact |

The heterogeneous adapter uses the global-average-pooled ResNet stage-3
representation (`64d`) and the DeiT CLS pre-logits representation (`192d`).
The official normalized linear projections map both to `128d`. No spatial
feature interpolation is required.

The checkpoint stores both the student and CRD criterion/memory-bank state so
the result is self-contained. Base optimization and input settings are fixed
in `methods/README.md`.

## Independent dataset entry points

| Dataset | Epochs | Batch | Warm-up | Entry point |
|---|---:|---:|---:|---|
| CIFAR-100 | 300 | 128 | 20 | [`cifar100/train.py`](cifar100/train.py) |
| Flowers-102 | 200 | 64 | 5 | [`flowers102/train.py`](flowers102/train.py) |
| Chaoyang | 100 | 64 | 5 | [`chaoyang/train.py`](chaoyang/train.py) |

Each dataset directory contains its own README and can be run without a batch
runner. Multi-method runners invoke these same wrappers and preserve separate
CRD checkpoints, memory-bank state, summaries, and output directories.
