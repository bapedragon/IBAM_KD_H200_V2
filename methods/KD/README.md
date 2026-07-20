# Standard logit KD

Standard KD transfers only the final class distribution, so the ResNet56 and
DeiT-Ti spatial feature grids do not need to match.

## Objective

```text
L = (1 - alpha) * CE + alpha * T^2 * KL(student/T, teacher/T)
T = 4.0
alpha = 0.9
```

`T` and `alpha` are fixed V2 implementation choices because the draft does not
specify them. The base student protocol is recorded in `methods/README.md`.
The teacher receives the shared student view bilinearly resized to 32x32.

## Output

Each run directory contains:

```text
student_best.pt
student_latest.pt
summary.json
```

Both checkpoints include the student state, epoch, Top-1, fixed-teacher
metadata, method name, and full command configuration.

## Independent dataset entry points

| Dataset | Epochs | Batch | Warm-up | Entry point |
|---|---:|---:|---:|---|
| CIFAR-100 | 300 | 128 | 20 | [`cifar100/train.py`](cifar100/train.py) |
| Flowers-102 | 200 | 64 | 5 | [`flowers102/train.py`](flowers102/train.py) |
| Chaoyang | 100 | 64 | 5 | [`chaoyang/train.py`](chaoyang/train.py) |

Each dataset directory contains its own README with the locked protocol,
standalone timing/full commands, and output contract. Multi-method runners call
these same entry points and do not contain a separate KD implementation.
