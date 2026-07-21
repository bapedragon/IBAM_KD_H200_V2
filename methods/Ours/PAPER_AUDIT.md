# Ours paper/source audit

This audit separates (1) values stated in V3/ALG, (2) details present only in
the supplied Ours model source, and (3) unavoidable reproduction choices.
The dataset-specific base protocols are intentionally documented elsewhere.

## Confirmed by V3 and the cited ALG paper

| Item | Implemented behavior |
|---|---|
| Total objective | `CE + beta(e) * [lambda*L_fuse + (1-lambda)*L_align]` |
| Loss balance | `lambda=0.5` |
| Alignment/fusion losses | MSE to the frozen CNN feature |
| Student aggregation | features from all `N=12` DeiT blocks with learned convex weights, uniformly initialized |
| Channel/grid alignment | stage-specific `1x1` projection; V3 describes bilinear resize to the teacher-stage resolution |
| Fusion | channel attention, deformable spatial attention, and grid-preserving convolutional cross-attention |
| Kernels | deformable spatial kernel `5x5`; Q/K/V projections `1x1` |
| Adaptive beta | ALG `beta=2.5`, `tau=-0.02`, 50-epoch loss smoothing/differentiation and 50-epoch derivative smoothing |
| Stop rule in researcher code | after a 20-epoch controller warm-up, stop when the computed smoothed derivative is strictly `> tau`; the crossing epoch is the last guided epoch, then CE only |
| Teacher/inference | teacher frozen during training; teacher and Ours modules removed at inference |

The active controller ports the three branches shown in the researcher code:
`e<=50`, `50<e<100`, and `e>=100`. It records the complete combined feature
loss per epoch, does not evaluate the stop rule before epoch 20, and adds no
descent-first guard. These implementation details are stronger evidence than
the earlier boundary assumptions made from the equations alone.

## Fixed by the supplied Ours model source, not stated numerically in V3

The supplied file is identified by SHA-256
`8649078970b93d750a956994611b65cdec0c24f907d35d86f29d635e8a3b8624`.

| Item | Source-derived setting |
|---|---|
| CNN stages | ResNet56 stages 1/2/3 |
| Aggregation | one learned 12-block mixture per CNN stage |
| Attention heads | 4 |
| Channel-attention reduction | 16 |
| Cross-attention dropout | 0 |
| Multi-stage loss reduction | sum of per-stage mean-squared errors |

The delivered file is a pycls wrapper whose behavior also depends on a config
that was not supplied. The standalone integration fixes
`ENABLE_INTER=True`, `INTER_TRANSFORM=linear`, and `ENABLE_LOGIT=False` so the
executed objective matches V3 Eq. (4). Thus the active Ours feature path is
source-faithful, but the repository file is not a byte-for-byte copy of the
pycls wrapper.

There is one paper/source inconsistency. V3 says to resample the student to the
teacher resolution, while both the originally supplied source and the later
researcher screenshot resize both tensors to the larger grid. The active
researcher-synchronized implementation therefore uses `--grid-resize-mode
larger` (`32/16/14`) for Chaoyang. The former paper-wording `32/16/8` run is
retained only as a historical diagnostic.

## Ours-specific reproduction choices not fixed by either paper

| Choice | Repository decision | Reason |
|---|---|---|
| Signal observed by ALG | epoch-average `0.5*L_align + 0.5*L_fuse` | Confirmed by the researcher trainer: the complete `loss_inter` returned by `guidance_loss()` is passed to the controller. |
| Initial derivative boundary | no stop before controller warm-up 20; afterward use the researcher's strict `smoothed_derivative > tau` test with no descent guard | Confirmed by the researcher controller screenshots. |
| Missing pycls config | feature guidance on, linear projection, logit KD off | The config file was not delivered; V3 Eq. (4) contains CE plus the two feature losses and no logit-KL term. |
| Teacher input size | fixed V2 32 x 32 teachers for all datasets | ALG confirms the low-resolution CNN path; the same fixed teacher is reused across compared methods. |
| Evaluation geometry | direct resize of the full image to `224 x 224`, then shared-view bilinear downsampling to `32 x 32` for the teacher | This follows the supplied/public locality-guidance loader. |
| Native teacher audit | compare direct `32 x 32` Top-1 with checkpoint metadata | This reproduces the teacher's own evaluation recipe and is the full-run safety gate. |
| Shared-view teacher diagnostic | report Top-1 after source-faithful `224 -> 32` two-step resampling | This input is used for feature guidance. Its classification Top-1 is diagnostic only because it is not the teacher checkpoint's native evaluation recipe. |

The current V2 teacher checkpoints were trained and verified at 32 pixels.
Every run reports `[TEACHER_NATIVE_AUDIT]` to detect checkpoint/data
integration regressions before a long H200 job and `[TEACHER_SHARED_VIEW]` to
record the actual guidance input. Ours wrappers use `--eval-resize-mode
direct`; the generic-KD center-crop path remains unchanged for historical
result compatibility.

## Shared experiment choices (not Ours-specific)

Epochs and best-checkpoint selection belong to the shared comparison protocol.
The researcher-synchronized Chaoyang Ours profile uses 300 epochs, train batch
64, eval batch 200, a 20-epoch LR warm-up, FP32, seed 1, and public LG
augmentation/regularization. Batch 64 comes from the supplied researcher
configuration; changing it to 128 is a separate protocol rather than a repeat
seed.
CIFAR-100 and Flowers-102 retain their earlier profiles until separately
audited and must not be described as using this exact Chaoyang ALG base.

## Full-run gate

1. Run the dataset wrapper with `--timing-run`.
2. Confirm dataset/split, finite loss, exact ALG parameters, epoch timing,
   `[TEACHER_NATIVE_AUDIT]`, the diagnostic `[TEACHER_SHARED_VIEW]`, and the
   wrapper-specific target grids. Chaoyang must print `32/16/14` and
   `metric=L_feature_combined`.
3. If the teacher audit passes, run the dataset-specific full command.
4. Keep the generated `summary.json`, which records the complete loss,
   derivative, beta, stop-epoch, and aggregation-weight histories.
