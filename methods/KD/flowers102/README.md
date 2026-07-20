# KD: Flowers-102 / DeiT-Ti

`train.py` independently trains a DeiT-Ti student with standard logit KD from
the fixed Flowers-102 ResNet56 teacher. The batch runners are convenience
tools; this entry point does not depend on them.

## Locked protocol

| Setting | Value |
|---|---:|
| Split | official train + val / official test |
| Epochs | 200 |
| Batch size | 64 |
| Warm-up | 5 epochs |
| Student / teacher input | 224 x 224 / 32 x 32 |
| Optimizer | AdamW |
| Initial LR / weight decay | `5e-4` / `0.05` |
| Label smoothing / seed | `0.1` / `42` |

The teacher receives the same augmented student view after bilinear resize to
32 x 32. KD uses temperature `4.0` and weight `0.9`; see
[`../README.md`](../README.md) for the exact objective.

## Run

```bash
python methods/KD/flowers102/train.py --timing-run --num-workers 4
python methods/KD/flowers102/train.py --student-epochs 200 --num-workers 4 --run-name kd_flowers102_deit_ti_200ep --output-dir /app/output
```

Each run writes `student_best.pt`, `student_latest.pt`, and `summary.json` in
its run directory.
