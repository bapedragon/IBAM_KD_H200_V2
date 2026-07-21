# Ours: Flowers-102 / DeiT-Ti

- Teacher: fixed V2 32 x 32 ResNet56 checkpoint selected by the manifest
- Student: DeiT-Ti from scratch
- Split: official `train + val` for training; official `test` for evaluation
- Base protocol: 200 epochs, batch 64, AdamW `5e-4`, minimum LR `0`,
  weight decay `0.05`, 5-epoch warm-up, cosine decay
- Input/recorded choices: 224 pixels, label smoothing `0.1`, seed `42`
- Loss: `CE + beta(e) * (0.5 * L_fuse + 0.5 * L_align)`
- Grid: V3 teacher-resolution policy, producing stage targets `32/16/8`
- Adaptive beta: exact ALG equations with `beta=2.5`, `tau=-0.02`, two
  50-epoch smoothing stages; `L_align` is the recorded controller signal
- Working-paper comparison target: `70.31%` Top-1

The fixed Flowers teacher was trained at 32 x 32. The runtime audit verifies
its manifest hash and preprocessing integration before student training.

Timing run:

```bash
python methods/Ours/flowers102/train.py --timing-run --num-workers 4
```

Full run only after the timing log and teacher audit pass:

```bash
python methods/Ours/flowers102/train.py --student-epochs 200 --num-workers 4 --run-name ours_flowers102_deit_ti_papergrid_200ep --output-dir /app/output
```

The 200-epoch dataset protocol is retained intentionally; the draft's single
300-epoch statement is marked for correction. See
[`../PAPER_AUDIT.md`](../PAPER_AUDIT.md) for confirmed and unresolved settings.
