# Ours: Flowers-102 / DeiT-Ti

- Teacher: fixed V2 32 x 32 ResNet56 checkpoint selected by the manifest
- Student: DeiT-Ti from scratch
- Evaluation: direct full-image resize to `224 x 224`, matching the supplied
  locality-guidance loader
- Current split: official train+val `2,040` for training and official test
  `6,149` for evaluation and best-checkpoint selection
- Paper-first training protocol: 300 epochs, train/eval batch `128/200`, AdamW
  `5e-4`, minimum LR `5e-6`, weight decay `0.05`, 20-epoch warm-up
  from factor `0.001`, cosine decay
- Input/regularization: 224 pixels, public LG strong augmentation,
  label smoothing `0.0`, drop path `0.1`, seed `1`, FP32
- Loss: `CE + beta(e) * (0.5 * L_fuse + 0.5 * L_align)`
- Grid: supplied-source larger-grid policy, producing targets `32/16/14`
- Adaptive beta: supplied Ours/researcher controller with `beta=2.5`, `tau=-0.02`, window
  50 and controller warm-up 20. The complete combined feature loss is observed;
  afterward strict `smoothed_derivative > tau` disables guidance permanently.
- Working-paper comparison target: `70.31%` Top-1

## Selected repository result

The consolidated table currently selects the fully imported
`researcher_sync_v1_300ep_seed1` run: train/eval batch `64/200`, 300 epochs,
official train+val/test, best epoch 251, **74.81% best Top-1**, and 74.21%
last-epoch Top-1. Batch 64 matches the shared CIFAR-100 researcher-sync
training setting requested for this comparison. Its checkpoint and summary
are stored under `results/Ours/flowers102/researcher_sync_v1_300ep_seed1/`.

The method-separated batch-128 entry point documented below was also run and
reached 72.78%, but it is retained as an auxiliary diagnostic rather than
replacing the selected batch-64 result.

The fixed Flowers teacher was trained at 32 x 32. The runtime audit verifies
its manifest hash and preprocessing integration before student training.

Timing run:

```bash
python methods/Ours/flowers102/train_official_split.py --timing-run --num-workers 4
```

Full run only after the timing log and teacher audit pass:

```bash
python methods/Ours/flowers102/train_official_split.py --num-workers 4 \
  --run-name ours_flowers102_deit_ti_trainval_test_300ep_seed1 \
  --output-dir /app/output
```

The current priority is Ours paper for shared training values, supplied Ours
source for module behavior, and ALG/LG only for settings absent from both.
Therefore batch 128 comes from the Ours paper, while larger-grid resizing,
all-block aggregation, and the combined guidance signal come from the supplied
Ours implementation. The older `train.py` entry point preserves the earlier
researcher-sync-v1 behavior for provenance only. It is not the current
active train+val/test entry point. Earlier results are never overwritten. See
[`../RESEARCHER_SYNC.md`](../RESEARCHER_SYNC.md) for the synchronized
controller and base protocol.
