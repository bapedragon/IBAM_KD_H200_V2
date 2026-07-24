# Adaptive Locality Guidance (ALG)

The active ALG implementation is method-isolated from `Ours`. It runs the
published adaptive controller on the official LG training base implemented in
[`methods/LG`](../LG).

## Locked method

- LG source: `lkhl/tiny-transformers` commit
  `d2165f74049c906b0afc9f957491960fb3c0cc8b`.
- Student features: DeiT-Ti blocks `[0, 6, 11]`.
- Teacher features: ResNet56 stages `[0, 1, 2]`.
- Alignment: learned stage-wise `1 x 1` projections, bilinear resize to the
  larger grid, and the sum of stage mean MSE.
- Training loss while active: `CE + 2.5 * LG`.
- Adaptive controller: ALG paper equations, window `50`, threshold `-0.02`,
  `smoothed derivative >= threshold` stop boundary, and no extra controller
  warm-up. The optimizer retains the official 20-epoch LR warm-up.
- DeiT classifier head: zero initialized as in the official LG source.
- AdamW: official four parameter groups; biases, one-dimensional parameters,
  `cls_token`, and `pos_embed` receive zero weight decay.
- Batch `128`, eval batch `200`, 300 epochs, seed `1`, FP32, direct bicubic
  224-pixel student view and bilinear 32-pixel teacher view.

The standard wrappers reject the historical researcher normalization,
strict-`>` stop comparison, controller warm-up 20, `draft_common`, and
Ours-matched optimizer/data settings.

Primary sources:

- [ALG paper DOI](https://doi.org/10.1109/TNNLS.2024.3515076)
- [Official LG paper](https://arxiv.org/abs/2207.10026)
- [Official LG repository](https://github.com/lkhl/tiny-transformers)

## Entry points

```bash
python methods/ALG/cifar100/train.py
python methods/ALG/flowers102/train.py
python methods/ALG/chaoyang/train.py
python methods/ALG/cub200/train.py
```

CUB-200 is a protocol transfer, not a result claimed by either source paper:
it uses the authors' official CUB train/test split, the repository's shared
scratch ResNet56 teacher, and the otherwise unchanged LG/ALG mechanics.

Historical noncanonical diagnostics are labeled under [`legacy`](legacy) and
are excluded from active runners.
