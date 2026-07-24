# Table 4 follow-up: independent global permutations for K and V

This isolated follow-up implements the first experiment requested after the
completed Table 4 grid-permutation control. It keeps that run unchanged and
changes only the tensor supplied to attention value `V`.

## Intervention

At each stage, after the same larger-grid resize used by the verified
CIFAR-100 Ours run:

```text
Q = unpermuted aligned student feature
K = P_K(teacher feature)
V = P_V(teacher feature), where P_K != P_V
L_align target = P_K(teacher feature)
L_fuse target  = P_K(teacher feature)
```

`P_K` preserves the previous experiment's seeds `1, 2, 3`. `P_V` uses
independent seeds `1001, 1002, 1003`. Both are fixed across samples, channels,
batches, and epochs. Only spatial `H×W` positions are rearranged; a complete
channel vector moves together.

This avoids the cancellation identity that applies when K and V receive the
same permutation:

```text
softmax(Q(PK)^T)PV = softmax(QK^T)V
```

No local patch permutation is included here; that is a separate second
experiment.

## Locked protocol

Every non-intervention setting is identical to
[`../table4_grid_permutation/train_cifar100.py`](../table4_grid_permutation/train_cifar100.py):

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
python methods/Ours/table4_kv_independent_permutation/train_cifar100.py \
  --student-epochs 300 \
  --num-workers 4 \
  --run-name table4_kv_independent_cifar100_300ep_seed1_k1_v1001 \
  --output-dir /app/output
```

The startup log must show one `K` and one `V` line per stage, with different
seeds and different stored permutation buffers. The final report prints best
Top-1 and checkpoint paths through the shared Ours training pipeline.

## Completed result

The 300-epoch run completed with best Top-1 **81.00%** at epoch 298 and
last Top-1 `80.87%`. Checkpoint inspection confirmed independent K/V seeds
`1/1001` and full permutations at all three stages.
