# Ours: grid-preserving CNN-to-ViT distillation

This folder runs the supplied Ours module with the repository's frozen
ResNet56 teachers and a DeiT-Ti student. The integration separates settings
confirmed by the working paper/source from reproduction choices that are not
available in the supplied materials. See [`PAPER_AUDIT.md`](PAPER_AUDIT.md)
for the evidence matrix.

`ours.py` is a standalone PyTorch port rather than a byte-for-byte copy of the
original pycls wrapper. Its aggregation, projection, attention blocks, and MSE
calculation preserve the supplied implementation. Grid sizing is an explicit
runtime policy because the supplied source and V3 disagree.
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

## Feature path and paper-grid decision

- frozen ResNet56 teacher stages 1/2/3
- patch-grid outputs from all 12 DeiT-Ti blocks
- one learned convex 12-block mixture per CNN stage
- stage-specific `1 x 1` channel projection
- bilinear resizing of the student representation to the teacher-stage grid,
  following V3
- channel attention and `5 x 5` deformable spatial attention
- four-head convolutional cross-attention with `1 x 1` Q/K/V
- teacher and Ours module discarded at inference

V3 explicitly says that the student representation reaches the teacher grid
after bilinear resampling. The fixed ResNet56 stages are verified as
`32 x 32`, `16 x 16`, and `8 x 8`; therefore table-targeted runs use
`--grid-resize-mode teacher`. The supplied source instead takes the larger of
the student and teacher grids, producing `32/16/14`. That behavior is retained
as `--grid-resize-mode larger` only for compatibility checks. Results from the
two modes must not be mixed.

## Adaptive beta from ALG

The default controller now implements ALG Eqs. (10)-(19), not a plateau
proxy. It uses `beta=2.5`, `tau=-0.02`, and a 50-epoch window in both
smoothing steps. Guidance remains active while the twice-smoothed loss
derivative is below `tau`; when it reaches `tau`, that epoch is the last
guided epoch and all subsequent epochs use CE only.

V3 does not state which of its two feature losses should be observed by the
ALG controller. This implementation observes `L_align`, because it is the
direct CNN/ViT feature distance corresponding most closely to ALG's `L_LG`.
At epoch 1, where ALG's published expression has no previous loss, the raw
derivative is initialized to zero and cannot stop guidance. These two boundary
decisions are explicitly saved in logs/checkpoints. `manual_stop` remains
available only for controlled diagnostics.

## Student base protocols

The current working draft states that every student uses 300 epochs, batch
size 128, and a 20-epoch warm-up. CIFAR-100 already follows that statement.
Chaoyang Ours now also uses it explicitly so the measured result can be
compared with the draft's `86.35%` table cell. Flowers-102 still retains the
earlier dataset-specific profile and therefore must not be described as a
strict reproduction of the draft's uniform protocol.

| Dataset | Epochs | Batch | Optimizer | LR / min LR | Weight decay | Warm-up | Schedule |
|---|---:|---:|---|---:|---:|---:|---|
| CIFAR-100 | 300 | 128 | AdamW | `5e-4` / `0` | `0.05` | 20 | Cosine |
| Flowers-102 | 200 | 64 | AdamW | `5e-4` / `0` | `0.05` | 5 | Cosine |
| Chaoyang | 300 | 128 | AdamW | `5e-4` / `0` | `0.05` | 20 | Cosine |

All use 224-pixel student inputs, label smoothing `0.1`, AMP, seed `42`, no
external student pretraining, the established dataset splits, and best Top-1
checkpoint reporting. Augmentation is the repository's common
`RandomResizedCrop(scale=0.8..1.0) + RandomHorizontalFlip` pipeline. These
regularization/checkpoint choices are recorded experiment settings, not
claimed as working-paper Ours specifications.

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
python methods/Ours/cifar100/train.py --student-epochs 300 --num-workers 4 --run-name ours_cifar100_deit_ti_papergrid_300ep --output-dir /app/output
python methods/Ours/flowers102/train.py --student-epochs 200 --num-workers 4 --run-name ours_flowers102_deit_ti_papergrid_200ep --output-dir /app/output
python methods/Ours/chaoyang/train.py --student-epochs 100 --num-workers 4 --run-name ours_chaoyang_deit_ti_papergrid_100ep --output-dir /app/output
```

For a manual diagnostic stop epoch, use:

```text
--beta-schedule manual_stop --guidance-stop-epoch <LAST_GUIDED_EPOCH>
```

Every epoch prints total/CE/alignment/fusion loss, beta, guidance state,
train/validation/best Top-1, learning rate, epoch time, and projected duration.
Failures end with `[FATAL]`; successful completion ends with `[DONE]`.
