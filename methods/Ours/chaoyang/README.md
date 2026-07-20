# Ours: Chaoyang / DeiT-Ti

- Data: official mounted dataset under `/app/data/chaoyang`
- Teacher: fixed V2 32 x 32 ResNet56 checkpoint selected by the manifest
- Student: DeiT-Ti from scratch
- Base protocol: 100 epochs, batch 64, AdamW `5e-4`, minimum LR `0`,
  weight decay `0.05`, 5-epoch warm-up, cosine decay
- Input/recorded choices: 224 pixels, label smoothing `0.1`, seed `42`
- Loss: `CE + beta(e) * (0.5 * L_fuse + 0.5 * L_align)`
- Adaptive beta: exact ALG equations with `beta=2.5`, `tau=-0.02`, two
  50-epoch smoothing stages; `L_align` is the recorded controller signal
- Working-paper comparison target: `86.35%` Top-1

The fixed Chaoyang teacher was trained at 32 x 32. The runtime audit verifies
its manifest hash and preprocessing integration before student training.

Timing run:

```bash
python methods/Ours/chaoyang/train.py --timing-run --num-workers 4
```

Full run only after the timing log and teacher audit pass:

```bash
python methods/Ours/chaoyang/train.py --student-epochs 100 --num-workers 4 --run-name ours_chaoyang_deit_ti_100ep --output-dir /app/output
```

Raw measured accuracy is retained; no teacher-gap correction is applied by
code. The 100-epoch dataset protocol is intentional. See
[`../PAPER_AUDIT.md`](../PAPER_AUDIT.md).
