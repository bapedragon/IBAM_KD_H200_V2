# Ours: Chaoyang / DeiT-Ti

- Data: official mounted dataset under `/app/data/chaoyang`
- Teacher: fixed V2 32 x 32 ResNet56 checkpoint selected by the manifest
- Student: DeiT-Ti from scratch
- ALG-matched base protocol: 300 epochs, batch 128, AdamW `5e-4`, minimum LR
  `5e-6`, weight decay `0.05`, 20-epoch warm-up from `5e-7`, cosine decay
- Public LG/ALG regularization: FP32, seed `1`, label smoothing `0`, drop path
  `0.1`, ImageNet normalization, color jitter `0.4`, RandAugment
  `rand-m9-mstd0.5-inc1`, random erasing `0.25`/pixel, bicubic interpolation
- Evaluation geometry: direct full-image resize to `224 x 224` (no center
  crop), then shared-view bilinear downsampling to `32 x 32` for the teacher
- Loss: `CE + beta(e) * (0.5 * L_fuse + 0.5 * L_align)`
- Grid: supplied-source larger-grid policy. Teacher/student tensors are
  bilinearly resized to the larger grid, producing `32 x 32`, `16 x 16`, and
  `14 x 14` targets.
- Adaptive beta: exact ALG equations with `beta=2.5`, `tau=-0.02`, two
  50-epoch smoothing stages; `L_align` is the recorded controller signal. The
  one-way stop is armed only after first observing a derivative below `tau`.
- Working-paper comparison target: `86.35%` Top-1

The fixed Chaoyang teacher was trained and evaluated by directly resizing the
original image to 32 x 32. `[TEACHER_NATIVE_AUDIT]` reproduces that comparable
path and must stay within the default 5 pp threshold. The source-faithful Ours
feature path instead resizes the shared student view from 224 to 32; its
`[TEACHER_SHARED_VIEW]` classification Top-1 is recorded as a diagnostic and
is not compared as if it were the native checkpoint recipe.

Timing run:

```bash
python methods/Ours/chaoyang/train.py --timing-run --num-workers 4
```

Full run only after the timing log and teacher audit pass:

```bash
python methods/Ours/chaoyang/train.py --student-epochs 300 --batch-size 128 --warmup-epochs 20 --num-workers 4 --run-name ours_chaoyang_deit_ti_algbase_sourcegrid_300ep_seed1 --output-dir /app/output
```

Raw measured accuracy is retained; no teacher-gap correction is applied by
code. This wrapper uses the standalone ALG run's complete base configuration,
then replaces the ALG-only feature objective with the delivered Ours module
and the paper/researcher-confirmed 0.5/0.5 objective. See
[`../PAPER_AUDIT.md`](../PAPER_AUDIT.md).
