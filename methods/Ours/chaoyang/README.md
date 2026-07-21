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

## Protocol decision and run lineage

The final comparison protocol is the audited public LG/ALG base implemented by
this wrapper. ALG and Ours must share this base; Ours changes only the feature
module/objective described above. A higher result from an older, different
base protocol must not be substituted into the final comparison.

| Run | Base / grid | Seed | Best Top-1 | Status |
|---|---|---:|---:|---|
| Earlier Ours, 100 epochs | common base, teacher grid `32/16/8` | 42 | 81.21% | historical diagnostic |
| Earlier Ours, 300 epochs | common base, teacher grid `32/16/8` | 42 | 81.11% | historical diagnostic |
| Audited ALG, 300 epochs | public LG/ALG base, ALG larger-grid path | 1 | 80.32% | current matched baseline |
| Audited Ours, 300 epochs | public LG/ALG base, source larger grid `32/16/14` | 1 | 77.14% | current matched Ours result; requires investigation |

The old common base used light `RandomResizedCrop+flip`, label smoothing
`0.1`, drop path `0`, AMP, `Resize(256)+CenterCrop(224)` evaluation, minimum
LR `0`, and seed `42`. The audited public LG/ALG base instead uses the strong
LG augmentation listed above, label smoothing `0`, drop path `0.1`, FP32,
direct-resize evaluation, minimum LR `5e-6`, and seed `1`. The two result
families are therefore not repeated runs of one protocol.

In the matched public-LG/ALG pair, ALG stops guidance at epoch 217 and reaches
80.32%, while Ours stops at epoch 234 and reaches 77.14%. This reverses the
working-paper ordering, so the Ours controller signal/feature-loss scale must
be audited before treating 77.14% as a validated reproduction. Running ALG
once under the old common base is permitted only as a controlled ablation to
measure the base-protocol effect; it is not the final ALG number.
