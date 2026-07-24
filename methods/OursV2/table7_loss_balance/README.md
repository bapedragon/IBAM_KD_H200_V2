# Ours V2 Table 7: loss-balance sensitivity

This folder isolates the CIFAR-100 loss-balance control from the Table 4
grid-permutation and token-space controls. The architecture is always the
position-aware Ours V2 `relative_position_v1` model. Only `lambda` changes in:

```text
L_total(e) = CE + beta(e) * [lambda * L_fuse + (1-lambda) * L_align]
```

For the requested `lambda=0` run, the weighted feature loss is exactly
`L_align`. The fusion block and its learned 2D relative-position parameters
remain instantiated, but `L_fuse` has zero weight and therefore contributes no
training signal. This is the alignment-only row defined by Table 7.

## Valid comparison

The direct Table 7 comparison is:

```text
Ours V2 relative_position_v1, lambda=0.5
vs.
Ours V2 relative_position_v1, lambda=0
```

Do not use either `grid_permutation_v1` or `token_space_v1` as the
`lambda=0.5` reference. Those variants change the attention mechanism and are
Table 4 controls. Do not use a pre-V2 Ours result as the reference either,
because it lacks the learned 2D relative-position bias.

## Locked protocol

- CIFAR-100, ResNet56 teacher, DeiT-Ti student
- 300 epochs, train/eval batch `64/200`
- AdamW, LR `5e-4`, minimum LR `5e-6`, weight decay `0.05`
- 20-epoch LR and controller warm-up
- ALG adaptive guidance: beta `2.5`, tau `-0.02`, window `50`
- larger-grid policy `32/16/14`, FP32, seed `1`
- Ours V2 learned head-specific 2D relative-position bias

## H200 full run

Run the paired full-data, two-epoch timing audit first:

```bash
python methods/OursV2/table7_loss_balance/run_cifar100.py \
  --timing-run --num-workers 4
```

This runs `lambda=0` followed by `lambda=0.5`, validates the two summaries,
prints the individual and combined 300-epoch estimates, and checks the
10-hour Pod limit. If it reports `PASS`, use:

```bash
python methods/OursV2/table7_loss_balance/run_cifar100.py \
  --full-run --num-workers 4
```

The verified paired full-run request is recorded in
[`H200_FULL_ISSUE.md`](H200_FULL_ISSUE.md).

The single-lambda entry point remains available for a split run:

```bash
python methods/OursV2/table7_loss_balance/train_cifar100.py \
  --lambda-value 0 \
  --num-workers 4 \
  --run-name table7_ours_v2_lambda_0_cifar100_300ep_seed1 \
  --output-dir /app/output
```

The startup audit must include:

```text
[OURS_V2_TABLE7_CONTROL] variant=relative_position_v1 reference_lambda=0.5 only_change=lambda lambda=0
[OURS_V2_TABLE7_LOSS] feature_loss=0*L_fuse+1*L_align ...
[OURS_V2] variant=relative_position_v1 ...
[OURS] ... lambda=0.0
```

The completed `summary.json` must report `method=OursV2`,
`ours_v2.variant=relative_position_v1`, `args.fusion_ratio=0.0`, and
`student_epochs=300`.

## Completed results

| Lambda | Best epoch | Best Top-1 | Last Top-1 | Gap to `lambda=0.5` |
|---:|---:|---:|---:|---:|
| 0 | 277 | **83.43%** | 83.42% | +0.59 pp |
| 0.5 | 273 | **82.84%** | 82.55% | reference |

Both checkpoints contain the `relative_position_v1` metadata and learned
head-specific 2D relative-position tables for all three fusion stages.
