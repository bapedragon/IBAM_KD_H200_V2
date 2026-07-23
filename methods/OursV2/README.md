# Ours V2: explicit 2D position-aware cross-attention

`methods/OursV2` is an isolated follow-up implementation.  It does not modify
or import-replace files under `methods/Ours`, so every historical Ours result
and entry point remains reproducible.

## Full V2

Ours V2 keeps the V1 aggregation, projection, deformable enhancement,
larger-grid resize, losses, adaptive controller, and training protocol.  It
changes only the fusion score:

```text
V1: score(p,q) = Q_p K_q^T / sqrt(d)
V2: score(p,q) = Q_p K_q^T / sqrt(d) + b_h(dx(p,q), dy(p,q))
```

`b_h` is one learned full 2D relative-displacement table per attention head and
per teacher stage.  The tables are initialized to zero.  Consequently, V2
starts from the exact V1 content-attention function and becomes position-aware
only as those tables learn.

The active V2 grids are `32x32`, `16x16`, and `14x14`.  The bias table supports
grids up to `32x32` by default.

## V2 controls

### Grid permutation

The V2 grid-permutation control jointly permutes teacher K/V content inside the
fusion block while retaining the learned 2D coordinate bias.  `L_align` and
`L_fuse` continue to use the original unpermuted teacher targets.  This avoids
the V1 control's confound where the position-wise MSE targets were permuted at
the same time as K/V.

### Token space

The V2 token-space control uses token-wise Linear Q/K/V projections and
content-only global attention.  It has no 2D relative-position term.  Because
`Conv1x1(B,C,H,W)` and `Linear(B,H*W,C)` are equivalent after reshaping, the
functional intervention is the removal of explicit 2D positional relations.

## Table 7 loss balance

The isolated
[`table7_loss_balance`](table7_loss_balance/README.md) entry point keeps the
base `relative_position_v1` architecture and all CIFAR-100 training settings
fixed while changing only `lambda`.  In particular, `lambda=0` is the
alignment-only control:

```text
L_feature = 0 * L_fuse + 1 * L_align
```

Its direct reference is base Ours V2 `relative_position_v1` at `lambda=0.5`,
not either Table 4 attention control and not a historical pre-V2 Ours result.

## CIFAR-100 commands

Run the full-data two-epoch timing checks first:

```bash
python methods/OursV2/cifar100/train.py \
  --timing-run --num-workers 4 \
  --output-dir /app/output/ours_v2_cifar100_timing

python methods/OursV2/grid_permutation/train_cifar100.py \
  --timing-run --num-workers 4 \
  --output-dir /app/output/ours_v2_grid_permutation_timing

python methods/OursV2/token_space/train_cifar100.py \
  --timing-run --num-workers 4 \
  --output-dir /app/output/ours_v2_token_space_timing
```

After all timing and teacher-native audits pass, run the three independent
300-epoch jobs:

```bash
python methods/OursV2/cifar100/train.py \
  --num-workers 4 \
  --output-dir /app/output/ours_v2_cifar100_300ep_seed1

python methods/OursV2/grid_permutation/train_cifar100.py \
  --num-workers 4 \
  --output-dir /app/output/ours_v2_grid_permutation_300ep_seed1

python methods/OursV2/token_space/train_cifar100.py \
  --num-workers 4 \
  --output-dir /app/output/ours_v2_token_space_300ep_seed1
```

Every checkpoint and summary is labeled `method=OursV2` and records its exact
variant.  Existing V1 checkpoints and accuracy values must not be reported as
V2 results.
