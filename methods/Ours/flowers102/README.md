# Ours: Flowers-102 / DeiT-Ti

- Teacher: fixed V2 32 x 32 ResNet56 checkpoint selected by the manifest
- Student: DeiT-Ti from scratch
- Evaluation: direct full-image resize to `224 x 224`, matching the supplied
  locality-guidance loader
- Split: official `train + val` for training; official `test` for evaluation
- Researcher-sync protocol: 300 epochs, train/eval batch `64/200`, AdamW
  `5e-4`, minimum LR `5e-6`, weight decay `0.05`, 20-epoch warm-up
  from factor `0.001`, cosine decay
- Input/regularization: 224 pixels, public LG strong augmentation,
  label smoothing `0.0`, drop path `0.1`, seed `1`, FP32
- Loss: `CE + beta(e) * (0.5 * L_fuse + 0.5 * L_align)`
- Grid: supplied-source larger-grid policy, producing targets `32/16/14`
- Adaptive beta: researcher controller with `beta=2.5`, `tau=-0.02`, window
  50 and controller warm-up 20. The complete combined feature loss is observed;
  afterward strict `smoothed_derivative > tau` disables guidance permanently.
- Working-paper comparison target: `70.31%` Top-1

The fixed Flowers teacher was trained at 32 x 32. The runtime audit verifies
its manifest hash and preprocessing integration before student training.

Timing run:

```bash
python methods/Ours/flowers102/train.py --timing-run --num-workers 4
```

Full run only after the timing log and teacher audit pass:

```bash
python methods/Ours/flowers102/train.py --num-workers 4 --run-name ours_flowers102_deit_ti_researcher_sync_300ep_seed1 --output-dir /app/output
```

The earlier 200-epoch, seed-42 run remains a historical result and is never
overwritten by this entry point. See [`../RESEARCHER_SYNC.md`](../RESEARCHER_SYNC.md)
for the synchronized controller and base protocol.
