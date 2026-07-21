# Ours: Chaoyang / DeiT-Ti

- Data: official mounted dataset under `/app/data/chaoyang`
- Teacher: fixed V2 32 x 32 ResNet56 checkpoint selected by the manifest
- Student: DeiT-Ti from scratch
- Working-draft comparison protocol: 300 epochs, batch 128, AdamW `5e-4`,
  minimum LR `0`, weight decay `0.05`, 20-epoch warm-up, cosine decay
- Input/recorded choices: 224 pixels, label smoothing `0.1`, seed `42`
- Evaluation geometry: direct full-image resize to `224 x 224` (no center
  crop), then shared-view bilinear downsampling to `32 x 32` for the teacher
- Loss: `CE + beta(e) * (0.5 * L_fuse + 0.5 * L_align)`
- Grid: V3 teacher-resolution policy. The student feature is bilinearly
  resampled onto the verified teacher stages `32 x 32`, `16 x 16`, and
  `8 x 8`; the supplied-source `14 x 14` final-stage rule is not used.
- Adaptive beta: exact ALG equations with `beta=2.5`, `tau=-0.02`, two
  50-epoch smoothing stages; `L_align` is the recorded controller signal
- Working-paper comparison target: `86.35%` Top-1

The fixed Chaoyang teacher was trained and evaluated by directly resizing the
original image to 32 x 32. `[TEACHER_NATIVE_AUDIT]` reproduces that comparable
path and must stay within the default 5 pp threshold. The source-faithful Ours
feature path instead resizes the shared student view from 224 to 32; its
`[TEACHER_SHARED_VIEW]` classification Top-1 is recorded as a diagnostic and
is not compared as if it were the native checkpoint recipe.

Timing run:

```bash
python methods/Ours/chaoyang/train.py --timing-run --grid-resize-mode teacher --num-workers 4
```

Full run only after the timing log and teacher audit pass:

```bash
python methods/Ours/chaoyang/train.py --student-epochs 300 --batch-size 128 --warmup-epochs 20 --grid-resize-mode teacher --num-workers 4 --run-name ours_chaoyang_deit_ti_papergrid_300ep_seed42 --output-dir /app/output
```

Raw measured accuracy is retained; no teacher-gap correction is applied by
code. This wrapper intentionally follows the current working draft's uniform
student statement (`300` epochs, batch `128`, 20-epoch warm-up) so its result
can be compared with the draft's `86.35%` Chaoyang cell. See
[`../PAPER_AUDIT.md`](../PAPER_AUDIT.md).
