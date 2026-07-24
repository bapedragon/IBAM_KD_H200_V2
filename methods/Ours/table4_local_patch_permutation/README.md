# Table 4 follow-up: local patch-grid permutation

This isolated follow-up implements the second experiment requested after the
completed global grid-permutation control.

## Intervention

After the same larger-grid resize used by the verified CIFAR-100 Ours run,
each teacher grid is divided into non-overlapping `2×2` windows. Spatial
positions are shuffled independently inside every window:

```text
teacher grid
  -> divide into non-overlapping 2x2 windows
  -> fixed independent HxW shuffle inside each window
  -> same locally permuted tensor for K, V, L_align, and L_fuse
```

No value crosses a window boundary. A full channel vector moves together, so
channels, samples, labels, and student tokens are unchanged. The three stages
use fixed seeds `1, 2, 3`; mappings remain fixed across all samples, batches,
and epochs. Each `2×2` mapping is a derangement, so all four positions move
within their own window. Because one tensor is passed to the standard Ours
fusion block, `K` and `V` receive exactly the same local permutation.

As in the completed same-K/V global control, applying one permutation to both
K and V leaves global cross-attention algebraically permutation-invariant.
This control therefore measures sensitivity to locally corrupted
position-wise MSE targets; it is not the independent-K/V experiment.

This is distinct from:

- the completed global control, which shuffles all `H×W` positions globally;
- the independent-K/V control, which gives K and V different global mappings.

## Locked protocol

Every non-intervention setting is identical to the verified CIFAR-100 Ours
protocol and the completed global control:

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
| Runtime | FP32, seed `1`, workers `4` |
| Grid | larger-grid policy: `32/16/14` |
| Objective | `CE + beta(e) * (0.5 L_fuse + 0.5 L_align)` |
| Adaptive guidance | beta `2.5`, tau `-0.02`, window `50`, gate warm-up `20` |
| Selection/reporting | best CIFAR-100 test Top-1 |

## H200 full run

```bash
python methods/Ours/table4_local_patch_permutation/train_cifar100.py \
  --student-epochs 300 \
  --num-workers 4 \
  --run-name table4_local_patch2_cifar100_300ep_seed1_permseed1 \
  --output-dir /app/output
```

Before accepting the run, the startup log must report stages `32×32`, `16×16`,
and `14×14`, window `2×2`, and non-zero `changed_positions`.

## Completed result

The 300-epoch run completed with best Top-1 **82.46%** at epoch 297 and
last Top-1 `82.41%`. Checkpoint inspection confirmed the `2x2` window size,
seed `1`, and complete within-window permutation buffers at all three stages.
