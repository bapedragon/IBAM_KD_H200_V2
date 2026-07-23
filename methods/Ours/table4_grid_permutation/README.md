# Table 4 control: grid-permuted teacher features

This isolated experiment fills the `Grid-permuted teacher features (c)` row
of Table 4. Its reference control is the verified CIFAR-100 Ours run at
`82.90%` Top-1. The normal Ours implementation and result are not modified.

## One-factor contract

Everything is locked to
[`../cifar100/train.py`](../cifar100/train.py), including the teacher,
student, data loader, optimizer, augmentation, adaptive controller, losses,
seed, checkpoint rule, and evaluation. The only change is:

```text
resized teacher grid
  -> one fixed spatial permutation per stage
  -> the same permuted tensor is used for K, V, L_align target, L_fuse target
```

The run-level permutation seed is `1`. Because the three active grids have
different position counts, that seed deterministically creates three stage
permutations for `32x32`, `16x16`, and `14x14`. They are fixed across every
sample, batch, and epoch. Tensor shapes, value multisets, operators, and
parameter counts remain unchanged.

## Locked protocol

| Item | Value |
|---|---|
| Dataset | CIFAR-100, train 50,000 / test 10,000 |
| Teacher | fixed V2 ResNet56, 32 px, selected Top-1 `71.91%` |
| Student | DeiT-Ti from scratch, 224 px |
| Epochs | 300 |
| Train / eval batch | 64 / 200 |
| Optimizer | AdamW, LR `5e-4`, min LR `5e-6`, weight decay `0.05` |
| LR schedule | 20-epoch warm-up, factor `0.001`, then cosine |
| Regularization | label smoothing `0`, drop path `0.1` |
| Augmentation | researcher-sync/public-LG strong augmentation |
| Runtime | FP32, seed `1`, workers `4` |
| Grid | larger-grid policy: `32/16/14` |
| Objective | `CE + beta(e) * (0.5 L_fuse + 0.5 L_align)` |
| Adaptive guidance | beta `2.5`, tau `-0.02`, window `50`, gate warm-up `20` |
| Selection/reporting | best CIFAR-100 test Top-1, matching the 82.90 control |

## H200 commands

Timing run over the full dataset (two epochs):

```bash
python methods/Ours/table4_grid_permutation/train_cifar100.py \
  --timing-run --num-workers 4 \
  --run-name table4_grid_permuted_cifar100_timing_2ep
```

Full Table 4 run after the timing log passes:

```bash
python methods/Ours/table4_grid_permutation/train_cifar100.py \
  --student-epochs 300 --num-workers 4 \
  --run-name table4_grid_permuted_cifar100_300ep_seed1_permseed1 \
  --output-dir /app/output
```

The completed 300-epoch result is:

| Best epoch | Best Top-1 | Last Top-1 | Full-Ours gap | Guidance stop |
|---:|---:|---:|---:|---:|
| 298 | **81.79%** | 81.61% | -1.11 pp | 137 |

The Table 4 cell is therefore `81.79`, not a projected value. It is `+1.99 pp`
above the draft token-space result (`79.80%`), so this exact intervention did
not produce the manuscript's expected collapse. The checkpoint and normalized
summary are imported at
`results/Ours/cifar100/table4_grid_permuted_researcher_sync_v1_300ep_seed1_permseed1/`;
the complete H200 log is under `results/run_logs/`.

## Expected audit lines

Startup must print all three lines below before full training is accepted:

```text
[TABLE4_GRID_PERMUTATION] enabled=True ...
[TABLE4_GRID_PERMUTATION] stage=1 grid=32x32 ...
[TABLE4_GRID_PERMUTATION] stage=2 grid=16x16 ...
[TABLE4_GRID_PERMUTATION] stage=3 grid=14x14 ...
```

If the final value does not collapse, report the measured value and revisit
the manuscript claim; do not replace it with an expected or fabricated value.
