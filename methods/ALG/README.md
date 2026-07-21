# Adaptive Locality Guidance (ALG)

This directory implements ALG for the fixed V2 ResNet56-to-DeiT-Ti setup.
ALG is **not** the repository's `Ours` method. It uses the original LG feature
matching objective and turns that objective off automatically when the
researcher-supplied controller's smoothed LG-loss derivative crosses its
threshold.

## Dataset targets

The ALG paper's Table II reports the following DeiT results on Chaoyang:

| Variant | Top-1 | Guidance stop epoch |
|---|---:|---:|
| Baseline | 80.65% | - |
| LG | 83.26% | never (all 300 epochs) |
| ALG | **83.50%** | **108** |

The local run reports both its Top-1 gap to `83.50%` and its observed stop-epoch
gap to `108`. The repository's fixed Chaoyang ResNet56 reaches `76.72%`, while
the original LG paper reports `78.12%` for its guidance model. Therefore this
is a carefully matched reproduction check, not a guaranteed bit-exact result.

Flowers-102 uses the same researcher-sync base and controller, the official
`train+val` / `test` split, and the working-paper references Vanilla `50.06%`,
LG `67.02%`, and ALG `68.54%`. Its dedicated entry point is
[`flowers102/train.py`](flowers102/train.py).

CIFAR-100 uses the same researcher-sync base and controller, the official
train/test split, and the working-paper ALG reference `81.98%`. Its dedicated
entry point is [`cifar100/train.py`](cifar100/train.py).

## Exact method path

LG behavior is ported from `lkhl/tiny-transformers` commit
`d2165f74049c906b0afc9f957491960fb3c0cc8b`:

- frozen ResNet56 stages `[0, 1, 2]`;
- DeiT-Ti blocks `[0, 6, 11]` (zero based);
- learned `1 x 1` projections `192 -> 16/32/64`;
- bilinear resizing of both features to the larger grid, producing
  `32 x 32`, `16 x 16`, and `14 x 14`;
- element-wise MSE at each stage, summed across all three stages.

The ALG controller is synchronized to the researcher implementation supplied
on 2026-07-21:

```text
L_total(e) = CE + 2.5 * L_LG(e), while guidance is active
stop after epoch e when smoothed_derivative(e) > -0.02
```

The controller records the complete three-stage LG loss, uses the exact
researcher three-case derivative calculation with a 50-epoch window, and does
not test for stopping before controller warm-up epoch 20. The comparison is
strictly `> tau`; there is no additional descent-first guard. The crossing
epoch is the final guided epoch and every later epoch is CE-only.

## Researcher-synchronized base protocol

| Setting | Value | Evidence |
|---|---:|---|
| Student | DeiT-Ti, scratch | ALG paper / LG public config |
| Student resolution | 224 x 224 | ALG paper |
| Teacher resolution | 32 x 32 | ALG paper / LG code |
| Epochs | 300 | ALG paper |
| Train / eval batch | 64 / 200 | researcher CIFAR config / supplied run config |
| Optimizer | AdamW | ALG paper |
| Initial / minimum LR | `5e-4` / `5e-6` | ALG paper / LG config |
| Weight decay | `0.05` | ALG paper |
| LR schedule | cosine | ALG paper |
| Warm-up | 20 epochs from factor `0.001` | ALG paper / LG config |
| Controller warm-up | 20 epochs | researcher trainer implementation |
| Drop path | `0.1` | LG public DeiT-Ti config |
| Label smoothing / Mixup / CutMix | `0` / off / off | LG public defaults |
| AMP | off (FP32) | LG public default |
| Seed | `1` | LG public default |
| Train drop-last | enabled | LG public loader |
| Eval geometry | direct bicubic resize to 224 | LG public loader |

The public strong augmentation path uses color-jitter argument `0.4`,
RandAugment `rand-m9-mstd0.5-inc1`, random erasing `p=0.25` in pixel mode,
bicubic interpolation, and ImageNet normalization. With RandAugment enabled,
the installed timm version realizes the secondary transform as RandAugment;
this is the original code path rather than a newly invented augmentation.

## Reproduction boundaries

- The ALG feature path is ported from cited LG code; the adaptive controller
  is synchronized to the researcher-supplied trainer screenshots.
- `timm==1.0.27` supplies DeiT-Ti instead of the older pycls implementation.
- Epoch 1 has no preceding loss in the derivative formula; it initializes the
  derivative to zero and cannot turn guidance off.
- Both best and latest checkpoints are saved. The Issue log reports best Top-1
  as the primary comparison and preserves latest Top-1 in `summary.json`.

## H200 commands

Full-data two-epoch timing check:

```bash
python methods/ALG/chaoyang/train.py --timing-run --num-workers 4
```

Full run after the timing checklist passes:

```bash
python methods/ALG/chaoyang/train.py --output-dir /app/output \
  --run-name alg_chaoyang_researcher_sync_300ep_seed1 --num-workers 4
```

Every epoch prints CE/LG/total loss, beta, both derivative values, guidance
state, train/test/best Top-1, learning rate, epoch time, and full-run estimate.
