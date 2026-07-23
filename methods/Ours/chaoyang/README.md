# Ours: Chaoyang / DeiT-Ti — researcher-synchronized run

This wrapper adapts the Ours code and configuration shown by the researcher to
the official mounted Chaoyang split. Dataset paths and the number of classes
are changed for Chaoyang; the transferable Ours/ALG behavior is kept intact.

## Locked protocol

| Item | Value | Source |
|---|---:|---|
| Teacher | ResNet56, frozen, `32 x 32` | repository checkpoint + researcher config |
| Student | DeiT-Ti from scratch, `224 x 224` | researcher config |
| Epochs | 300 | researcher config |
| Train / eval batch | 64 / 200 | researcher config |
| Optimizer | AdamW | researcher config |
| LR / minimum LR | `5e-4` / `5e-6` | researcher config |
| Weight decay | `0.05` | researcher config |
| LR warm-up | 20 epochs from factor `0.001` | researcher config |
| Schedule | cosine | researcher config |
| Label smoothing | `0` | public config default |
| Drop path | `0.1` | researcher config |
| Seed / precision | `1` / FP32 | public config defaults |
| Evaluation | direct resize to `224 x 224` | locality-guidance loader |

Public LG strong augmentation remains enabled: ImageNet normalization, color
jitter `0.4`, RandAugment `rand-m9-mstd0.5-inc1`, bicubic interpolation, and
random erasing `0.25` in pixel mode. Mixup and CutMix are zero by default.
EMA update period is also zero in the public config, so no EMA result is used.

## Ours feature path and loss

- all 12 DeiT block grids are aggregated into three learned stage mixtures;
- each stage is projected with a `1 x 1` convolution;
- teacher and student are both bilinearly resized to the larger grid, producing
  `32 x 32`, `16 x 16`, and `14 x 14`;
- the supplied CCC attention module produces the fused representation;
- `L_feature = 0.5*L_align + 0.5*L_fuse`;
- while guidance is active, `L_total = CE + 2.5*L_feature`;
- logit distillation is disabled.

The adaptive controller records the epoch mean of the complete `L_feature`,
not `L_align` alone. It uses `tau=-0.02`, a 50-epoch smoothing window, and a
20-epoch controller warm-up. It then stops permanently when the researcher
formula returns `smoothed_derivative > tau`; no descent-first guard is added.

## Run commands

Timing run (full data, two epochs, no persistent result required):

```bash
python methods/Ours/chaoyang/train.py --timing-run --num-workers 4
```

Full run after the timing log passes:

```bash
python methods/Ours/chaoyang/train.py --student-epochs 300 --batch-size 64 --warmup-epochs 20 --num-workers 4 --run-name ours_chaoyang_deit_ti_researcher_sync_300ep_seed1 --output-dir /app/output
```

Every checkpoint and `summary.json` records the complete arguments, teacher
hash, controller history, stop epoch, and feature aggregation weights. Older
Ours runs remain traceable under [`../legacy`](../legacy/README.md) and Git
commit `ee2dc55`; they are not repeated seeds of this protocol.

## Build-480 auxiliary result

The CIFAR-100-locked batch-64 variant completed 300 epochs at `81.11%` best
Top-1 (epoch 271), `80.22%` last Top-1, and guidance-stop epoch 192. It is
preserved as a protocol comparison under
`results/Ours/chaoyang/cifar100_locked_b64_v1_300ep_seed1/`, together with the
complete build-480 log/sequence summary. It does not replace the separately
log-verified `81.95%` researcher-sync result that is still awaiting its
individual checkpoint and per-run summary archive.
