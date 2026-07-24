# Official Locality Guidance (LG)

This package ports the public LG implementation at commit
`d2165f74049c906b0afc9f957491960fb3c0cc8b` into the repository's H200 runner
format. It is the independent base used by both static LG and canonical ALG;
it does not import Ours training settings.

The port preserves the official DeiT-Ti/ResNet56 mechanics:

- student blocks `[0, 6, 11]`, teacher stages `[0, 1, 2]`;
- learned `1 x 1` channel projections;
- bilinear resizing to the larger feature grid;
- summed stage-wise mean MSE with coefficient `2.5`;
- 224-pixel DeiT input and bilinear 32-pixel CNN guidance input;
- AdamW `5e-4`, minimum LR `5e-6`, weight decay `0.05`, cosine schedule,
  20-epoch LR warm-up from factor `0.001`, batch `128`, 300 epochs, FP32,
  seed `1`, and drop path `0.1`;
- zero-initialized DeiT classifier head;
- zero weight decay for biases, all one-dimensional parameters,
  `cls_token`, and `pos_embed`.

LG remains active for every training epoch. ALG imports this runtime and only
changes the guidance schedule.

Primary sources:

- [LG paper](https://arxiv.org/abs/2207.10026)
- [Official LG repository](https://github.com/lkhl/tiny-transformers)

## Entry points

```bash
python methods/LG/cifar100/train.py
python methods/LG/flowers102/train.py
python methods/LG/chaoyang/train.py
python methods/LG/cub200/train.py
```

The public LG repository did not publish a CUB-200 configuration. The CUB
entry is explicitly a transfer of the verified LG mechanics to the official
CUB-200-2011 train/test split and the shared scratch 32-pixel ResNet56 teacher.
