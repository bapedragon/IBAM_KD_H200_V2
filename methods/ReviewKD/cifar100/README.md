# ReviewKD: CIFAR-100 / DeiT-Ti

`train.py` independently trains a DeiT-Ti student with ReviewKD from the fixed
CIFAR-100 ResNet56 teacher. The batch runners are convenience tools; this
entry point does not depend on them.

## Locked protocol

| Setting | Value |
|---|---:|
| Epochs | 300 |
| Batch size | 128 |
| Warm-up | 20 epochs |
| Student / teacher input | 224 x 224 / 32 x 32 |
| Optimizer | AdamW |
| Initial LR / weight decay | `5e-4` / `0.05` |
| Label smoothing / seed | `0.1` / `42` |

ReviewKD connects ResNet stages 1/2/3 to DeiT blocks 3/7/11 with bilinear
spatial bridging and official-behavior ABF/HCL fusion. See
[`../README.md`](../README.md) for the feature-loss ramp and adapter details.

## Run

```bash
python methods/ReviewKD/cifar100/train.py --timing-run --num-workers 4
python methods/ReviewKD/cifar100/train.py --student-epochs 300 --num-workers 4 --run-name reviewkd_cifar100_deit_ti_300ep --output-dir /app/output
```

Each run writes `student_best.pt`, `student_latest.pt`, and `summary.json`.
The checkpoints include the trained ABF adapter state.
