# Ours: grid-preserving CNN-to-ViT distillation

This folder runs the supplied Ours module with the repository's frozen
ResNet56 teachers and a DeiT-Ti student. The integration separates settings
confirmed by the working paper/source from reproduction choices that are not
available in the supplied materials. See [`PAPER_AUDIT.md`](PAPER_AUDIT.md)
for the evidence matrix.

Table 4 attribution controls are isolated from the primary training entry
points. The fixed teacher-grid permutation control and its locked 82.90%-run
comparison contract live under
[`table4_grid_permutation/`](table4_grid_permutation/README.md).

The follow-up control in which key and value receive different fixed global
spatial permutations is isolated under
[`table4_kv_independent_permutation/`](table4_kv_independent_permutation/README.md).

Table 7 loss-balance controls are likewise isolated under
[`table7_loss_balance/`](table7_loss_balance/README.md). They inherit the same
82.90%-run protocol and vary only the convex loss-balance parameter `lambda`.

`ours.py` is a standalone PyTorch port rather than a byte-for-byte copy of the
original pycls wrapper. Its aggregation, projection, attention blocks, MSE
calculation, and larger-grid resize preserve the supplied implementation.
After the 2026-07-21 researcher-code audit, Chaoyang also follows the executable
larger-grid rule and therefore uses `32 x 32`, `16 x 16`, and `14 x 14`.
The unavailable pycls configuration is fixed to feature guidance enabled,
linear feature projection, and optional logit KD disabled, which matches V3
Eq. (4). The original source SHA-256 is recorded in every run.

## Implemented Ours objective

The executable now follows working-paper Eq. (4) directly:

```text
L_total(e) = CE + beta(e) * [0.5 * L_fuse + 0.5 * L_align]
```

- `L_align`: MSE between the projected/aligned student grid and CNN grid
- `L_fuse`: MSE between the fused grid and CNN grid
- guidance active: `beta(e) = beta_on`
- guidance inactive: `beta(e) = 0`, teacher/feature-module forward passes are
  skipped and training continues with CE only

The previous extra fixed multiplication
`CE + 1.0 * 2.5 * (...)` has been removed. `beta=2.5` is represented once as
`beta(e)`, following the ALG paper cited by V3.

## Feature path and source-grid decision

- frozen ResNet56 teacher stages 1/2/3
- patch-grid outputs from all 12 DeiT-Ti blocks
- one learned convex 12-block mixture per CNN stage
- stage-specific `1 x 1` channel projection
- bilinear resizing of both representations to the larger teacher/student
  grid at each stage, following the supplied Ours source
- channel attention and `5 x 5` deformable spatial attention
- four-head convolutional cross-attention with `1 x 1` Q/K/V
- teacher and Ours module discarded at inference

V3 describes resampling the student representation to the teacher grid, while
the supplied source and researcher screenshot explicitly resize both tensors
to the larger grid. The active researcher-synchronized path therefore uses
`--grid-resize-mode larger`, producing `32/16/14`. Earlier `32/16/8` results
remain historical diagnostics and are not repeated seeds of this protocol.

## Researcher-synchronized adaptive beta

The active controller is a direct port of the code shown by the researcher. It
uses `beta=2.5`, `tau=-0.02`, a 50-epoch window, and a controller warm-up of 20
epochs. Crucially, it records the complete guidance loss returned by
`guidance_loss()`—`0.5*L_align + 0.5*L_fuse`—rather than alignment loss alone.
After the warm-up, guidance is permanently disabled when the computed smoothed
derivative is strictly greater than `tau`. There is no extra descent-first
guard. The three early/middle/late derivative cases in the researcher code are
ported literally and covered by unit tests.

The former controller and Chaoyang profile are retained under `legacy/`, with
the full old executable available at Git commit `ee2dc55`. They must not be
mixed with researcher-synchronized results.

## Student base protocols

The active Ours entry points use the configuration supplied by the researcher.
This protocol family is deliberately separated from both the earlier
dataset-specific runs and the generic-KD epoch-only reruns.

| Dataset | Epochs | Batch | Optimizer | LR / min LR | Weight decay | Warm-up | Schedule |
|---|---:|---:|---|---:|---:|---:|---|
| CIFAR-100 (researcher sync) | 300 | 64 / eval 200 | AdamW | `5e-4` / `5e-6` | `0.05` | 20 | Cosine |
| Flowers-102 (researcher sync) | 300 | 64 / eval 200 | AdamW | `5e-4` / `5e-6` | `0.05` | 20 | Cosine |
| Chaoyang (researcher sync) | 300 | 64 / eval 200 | AdamW | `5e-4` / `5e-6` | `0.05` | 20 | Cosine |

All three are locked to the same audited LG/ALG base as the standalone ALG
run: FP32, seed
`1`, label smoothing `0`, drop path `0.1`, warm-up factor `0.001`, ImageNet
normalization, RandAugment `rand-m9-mstd0.5-inc1`, color jitter `0.4`, random
erasing `0.25` in pixel mode, bicubic interpolation, and drop-last training.
This makes each matched ALG/Ours comparison differ only in the Ours feature
module and objective. All students start without external pretraining and use
best Top-1 checkpoint reporting. Older 200/100-epoch files remain explicitly
labeled as historical results.

### Method-isolated Flowers comparison

[`flowers102/train_official_split.py`](flowers102/train_official_split.py) is
the active Flowers comparison path. It does **not** inherit ALG as a complete
training configuration. Precedence is fixed as follows:

1. Ours paper for the student protocol (`300` epochs, train batch `128`,
   AdamW, LR `5e-4`, weight decay `0.05`, cosine, LR warm-up `20`, 224 px);
2. supplied Ours source/researcher code for the Ours module, full guidance
   loss, larger-grid resizing, and 20-epoch controller gate;
3. ALG/public LG only for an item absent from both Ours sources.

It concatenates official Flowers train+val into `2,040` training images and
uses the official `6,149`-image test split for evaluation and best-checkpoint
selection, matching the ALG paper's dataset accounting. This path is
intentionally distinct from the older batch-64 researcher-sync diagnostic
described in the table above.

## Fixed V2 teacher and shared image geometry

All V2 teachers were trained and verified at 32 x 32. The same augmented
student image geometry is used for both branches: the 224 x 224 student tensor
is converted back to image space, bilinearly resized to 32 x 32, and normalized
with the teacher's ImageNet statistics. This matches the other V2 KD methods
and prevents independent crop/flip drift. The startup
`[TEACHER_NATIVE_AUDIT]` evaluates the checkpoint through its own direct-32px
recipe and is the full-run safety gate. `[TEACHER_SHARED_VIEW]` separately
reports the source-faithful 224-to-32 guidance view as a diagnostic.

At evaluation time, Ours directly resizes the full image to `224 x 224`
before making the shared 32-pixel teacher view. This matches the
supplied/public locality-guidance loader. It intentionally does not use the
generic-KD compatibility transform `Resize(256)+CenterCrop(224)`.

## H200 execution

Timing checks (full dataset, two epochs):

```bash
python methods/Ours/cifar100/train.py --timing-run --num-workers 4
python methods/Ours/flowers102/train.py --timing-run --num-workers 4
python methods/Ours/chaoyang/train.py --timing-run --num-workers 4
```

Conditional full runs after the timing log and teacher audit are accepted:

```bash
python methods/Ours/cifar100/train.py --student-epochs 300 --num-workers 4 --run-name ours_cifar100_deit_ti_sourcegrid_300ep --output-dir /app/output
python methods/Ours/flowers102/train.py --num-workers 4 --run-name ours_flowers102_deit_ti_researcher_sync_300ep_seed1 --output-dir /app/output
python methods/Ours/chaoyang/train.py --student-epochs 300 --batch-size 64 --warmup-epochs 20 --num-workers 4 --run-name ours_chaoyang_deit_ti_researcher_sync_300ep_seed1 --output-dir /app/output
```

For a manual diagnostic stop epoch, use:

```text
--beta-schedule manual_stop --guidance-stop-epoch <LAST_GUIDED_EPOCH>
```

Every epoch prints total/CE/alignment/fusion loss, beta, guidance state,
train/validation/best Top-1, learning rate, epoch time, and projected duration.
Failures end with `[FATAL]`; successful completion ends with `[DONE]`.
