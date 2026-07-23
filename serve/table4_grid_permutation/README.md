# Table 4 grid permutation — consolidated snapshot

This folder shows the complete **model-side execution path** of the completed
Table 4 grid-permutation control in one file:

- [`train_cifar100_full.py`](train_cifar100_full.py)

The file includes:

1. the supplied Ours modules;
2. student aggregation and channel projection;
3. teacher-grid resize;
4. deterministic stage-wise grid permutation;
5. unpermuted student query and permuted teacher key/value;
6. alignment and fusion MSE targets;
7. the exact CIFAR-100 protocol defaults and executable entry point.

Dataset loading, ResNet56/DeiT-Ti construction, checkpoint loading, optimizer,
scheduler, evaluation, and checkpoint saving still use the shared
`methods/Ours/core.py` infrastructure. Duplicating that infrastructure here
would create a second training implementation and could silently change the
experiment. The model/intervention logic itself is fully expanded here.

## Exact completed experiment

| Item | Value |
|---|---|
| Dataset | CIFAR-100 |
| Teacher | ResNet56, 32×32, fixed checkpoint |
| Student | DeiT-Ti, 224×224 |
| Epochs | 300 |
| Train/eval batch | 64 / 200 |
| Optimizer | AdamW |
| LR / minimum LR | 5e-4 / 5e-6 |
| Weight decay | 0.05 |
| Warm-up | 20 epochs, factor 0.001 |
| Schedule | Cosine |
| Seed | 1 |
| Grid resize | Larger grid: 32×32, 16×16, 14×14 |
| Permutation seed | 1 |
| Result | **81.79% Top-1** |
| Full Ours reference | 82.90% Top-1 |

## What is permuted

For stage `s`, a fixed permutation is generated with seed `1 + s` after grid
resize. It is fixed across samples, channels, batches, and epochs. The three
stages therefore use different permutations:

- stage 1: 32×32, seed 1;
- stage 2: 16×16, seed 2;
- stage 3: 14×14, seed 3.

Only teacher spatial positions are rearranged. A complete channel vector moves
together; channels, samples, labels, and student tokens are not shuffled. The
same permuted teacher tensor is used for:

- attention key `K`;
- attention value `V`;
- `L_align` target;
- `L_fuse` target.

The student feature remains unpermuted and forms query `Q`.

## Important interpretation

Applying the same spatial permutation to both `K` and `V` does not by itself
destroy global cross-attention, because
`softmax(Q(PK)^T)(PV) = softmax(QK^T)V` up to the corresponding column
permutation. The intervention mainly changes the position-wise MSE targets.
This explains why the completed control dropped only 1.11 percentage points
rather than collapsing below the token-space control.

This folder records exactly what was run; it does not alter the intervention
after seeing the result.

## H200 command

```bash
python serve/table4_grid_permutation/train_cifar100_full.py \
  --num-workers 4 \
  --output-dir /app/output \
  --run-name table4_grid_permuted_cifar100_300ep
```

The original modular source remains at
`methods/Ours/table4_grid_permutation/`.
