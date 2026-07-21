# Ours: CIFAR-100 / DeiT-Ti

- Teacher: fixed V2 32 x 32 ResNet56 checkpoint selected by the manifest
- Student: DeiT-Ti from scratch
- Evaluation: direct full-image resize to `224 x 224`, matching the supplied
  locality-guidance loader
- Researcher-synchronized config: 300 epochs, train batch 64, test batch 200,
  AdamW `5e-4`, minimum LR `5e-6`, weight decay `0.05`, warm-up factor
  `0.001`, 20-epoch warm-up, cosine decay
- Input/regularization: student 224 pixels, teacher 32 pixels, strong public
  LG augmentation, label smoothing `0`, drop path `0.1`, FP32, seed `1`
- Ours: all 12 student blocks, ResNet stages 1/2/3, learned stage mixtures,
  `1x1` projection/QKV, `5x5` deformable attention, four heads
- Grid: supplied-source larger-grid policy, producing targets `32/16/14`
- Loss: `CE + beta(e) * (0.5 * L_fuse + 0.5 * L_align)`
- Adaptive beta: researcher controller with `beta=2.5`, `tau=-0.02`, window
  50 and controller warm-up 20. The complete combined feature loss is observed;
  afterward strict `smoothed_derivative > tau` disables guidance permanently.
- Working-paper comparison target: `82.42%` Top-1

The teacher is already trained at 32 x 32. It receives the same crop/flip view
as the student after inverse student normalization, bilinear resize to 32, and
teacher normalization. Its manifest hash and runtime Top-1 are audited before
training.

Timing run:

```bash
python methods/Ours/cifar100/train.py --timing-run --num-workers 4
```

Full run only after the timing log and teacher audit pass:

```bash
python methods/Ours/cifar100/train.py --student-epochs 300 --num-workers 4 --run-name ours_cifar100_deit_ti_researcher_sync_300ep_seed1 --output-dir /app/output
```

See [`../PAPER_AUDIT.md`](../PAPER_AUDIT.md) for the paper/source/researcher
evidence split.
