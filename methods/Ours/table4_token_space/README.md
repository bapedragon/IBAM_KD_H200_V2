# Table 4 control: token-space cross-attention

This isolated experiment re-measures the working-paper
`Token-space cross-attention (b)` row under the exact protocol that produced
the verified CIFAR-100 full-Ours result of `82.90%`.

## One-factor contract

Everything remains locked to
[`../cifar100/train.py`](../cifar100/train.py): dataset, teacher, student,
learned aggregation, channel projection, channel/deformable enhancement,
losses, adaptive guidance, optimizer, augmentation, seed, checkpoint rule,
and evaluation.

Only the fusion Q/K/V implementation changes:

```text
Full Ours:
  BCHW grid -> Conv1x1 Q/K/V -> flatten -> multi-head cross-attention

Token-space control:
  BCHW grid -> flatten to BNC -> Linear Q/K/V -> multi-head cross-attention
```

The output token sequence is restored to BCHW so that the unchanged
position-wise fusion MSE can be evaluated. No teacher-grid permutation and no
positional encoding are introduced.

Feature guidance is evaluated only while `beta(e) > 0`. Once the unchanged ALG
controller disables guidance, the remaining epochs are CE-only, exactly as in
full Ours and the permutation controls.

## Locked protocol

| Item | Value |
|---|---|
| Dataset | CIFAR-100, train 50,000 / test 10,000 |
| Teacher | fixed V2 ResNet56, native 32 px, selected Top-1 `71.91%` |
| Student | DeiT-Ti from scratch, 224 px |
| Epochs | 300 |
| Train / eval batch | 64 / 200 |
| Optimizer | AdamW, LR `5e-4`, min LR `5e-6`, weight decay `0.05` |
| LR schedule | 20-epoch warm-up, factor `0.001`, then cosine |
| Regularization | label smoothing `0`, drop path `0.1` |
| Augmentation | researcher-sync/public-LG strong augmentation |
| Runtime | FP32, seed `1`, workers `4` |
| Grid resize | larger-grid policy: `32/16/14` |
| Objective | `CE + beta(e) * (0.5 L_fuse + 0.5 L_align)` |
| Adaptive guidance | beta `2.5`, tau `-0.02`, window `50`, gate warm-up `20` |
| Selection/reporting | best CIFAR-100 test Top-1 |

## Scientific caveat

A `1x1` convolution applied identically at every spatial position and a
fully connected layer applied independently to every flattened token have the
same parameter count and are mathematically equivalent after reshaping:

```text
Conv1x1(B,C,H,W) <=> Linear(B,H*W,C)
```

The included test verifies numerical equivalence after copying weights.
Therefore the manuscript's previous `79.80%` is not guaranteed to reproduce
from this stated substitution alone. This run must report the measured result;
it must not be adjusted toward the draft value.

## H200 commands

Timing run over the full dataset:

```bash
python methods/Ours/table4_token_space/train_cifar100.py \
  --timing-run \
  --num-workers 4 \
  --run-name table4_token_space_cifar100_timing_2ep
```

Full Table 4 run:

```bash
python methods/Ours/table4_token_space/train_cifar100.py \
  --student-epochs 300 \
  --num-workers 4 \
  --run-name table4_token_space_cifar100_300ep_seed1 \
  --output-dir /app/output
```

The startup log must contain:

```text
[TABLE4_CONTROL] ... only_change=grid_to_token_space_cross_attention
[TABLE4_TOKEN_SPACE] enabled=True ... token_qkv=Linear_on_flattened_BNC
```

## Completed result

The 300-epoch run completed with best Top-1 **83.12%** at epoch 290 and
last Top-1 `82.88%`. The checkpoint contains the token-space activation
marker and the expected Linear Q/K/V state.
