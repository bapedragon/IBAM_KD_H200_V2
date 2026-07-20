# MGD: Chaoyang / DeiT-Ti

`train.py` independently trains a DeiT-Ti student with MGD from the fixed
Chaoyang ResNet56 teacher. The batch runners are convenience tools; this entry
point does not depend on them.

## Locked protocol

| Setting | Value |
|---|---:|
| Data | official mounted train/test splits |
| Epochs | 100 |
| Batch size | 64 |
| Warm-up | 5 epochs |
| Student / teacher input | 224 x 224 / 32 x 32 |
| Optimizer | AdamW |
| Initial LR / weight decay | `5e-4` / `0.05` |
| Label smoothing / seed | `0.1` / `42` |

MGD aligns DeiT block-11 patch features to ResNet stage 3, masks channels, and
reconstructs the teacher map with the official generator. See
[`../README.md`](../README.md) for `alpha`, mask probability, and provenance.

## Run

```bash
python methods/MGD/chaoyang/train.py --timing-run --data-dir /app/data/chaoyang --num-workers 4
python methods/MGD/chaoyang/train.py --student-epochs 100 --data-dir /app/data/chaoyang --num-workers 4 --run-name mgd_chaoyang_deit_ti_100ep --output-dir /app/output
```

Each run writes `student_best.pt`, `student_latest.pt`, and `summary.json`.
The checkpoints include the MGD alignment and generator state.
