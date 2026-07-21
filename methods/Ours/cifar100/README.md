# Ours: CIFAR-100 / DeiT-Ti

- Teacher: fixed V2 32 x 32 ResNet56 checkpoint selected by the manifest
- Student: DeiT-Ti from scratch
- Base protocol: 300 epochs, batch 128, AdamW `5e-4`, minimum LR `0`,
  weight decay `0.05`, 20-epoch warm-up, cosine decay
- Input/recorded choices: 224 pixels, label smoothing `0.1`, seed `42`
- Ours: all 12 student blocks, ResNet stages 1/2/3, learned stage mixtures,
  `1x1` projection/QKV, `5x5` deformable attention, four heads
- Grid: V3 teacher-resolution policy, producing stage targets `32/16/8`
- Loss: `CE + beta(e) * (0.5 * L_fuse + 0.5 * L_align)`
- Adaptive beta: exact ALG equations with `beta=2.5`, `tau=-0.02`, two
  50-epoch smoothing stages; `L_align` is the recorded controller signal
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
python methods/Ours/cifar100/train.py --student-epochs 300 --num-workers 4 --run-name ours_cifar100_deit_ti_papergrid_300ep --output-dir /app/output
```

The ALG equations/values are paper-confirmed; selecting `L_align` as the
controller signal and the epoch-1 initialization are documented reproduction
choices. See [`../PAPER_AUDIT.md`](../PAPER_AUDIT.md).
