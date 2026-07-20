# CRD: Flowers-102 / DeiT-Ti

`train.py` independently trains a DeiT-Ti student with CRD from the fixed
Flowers-102 ResNet56 teacher. The batch runners are convenience tools; this
entry point does not depend on them.

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

CRD maps global-pooled ResNet stage-3 features and DeiT CLS pre-logits to a
shared 128-dimensional contrastive space. See [`../README.md`](../README.md)
for the official-code provenance, memory bank, and loss coefficients.

## Run

```bash
python methods/CRD/flowers102/train.py --timing-run --num-workers 4
python methods/CRD/flowers102/train.py --student-epochs 200 --num-workers 4 --run-name crd_flowers102_deit_ti_200ep --output-dir /app/output
```

Each run writes `student_best.pt`, `student_latest.pt`, and `summary.json`.
The checkpoints also include the CRD projection and memory-bank state.
