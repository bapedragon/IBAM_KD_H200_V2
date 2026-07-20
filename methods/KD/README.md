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
