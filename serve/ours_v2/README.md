# Ours V2 consolidated review snapshot

[`train_cifar100_full.py`](train_cifar100_full.py) expands the complete
model-side Ours V2 execution path in one reviewable file:

1. all-block learnable student aggregation;
2. stage-wise `1x1` channel projection;
3. larger-grid resize (`32/16/14`);
4. deformable channel and spatial attention;
5. convolutional `1x1` Q/K/V;
6. learned head-specific 2D relative-position bias;
7. global cross-attention;
8. `L_align` and `L_fuse`;
9. the locked CIFAR-100 protocol and Ours V2 artifact labels.

The key score is visible directly in the file:

```text
score(p,q) = Q_p K_q^T / sqrt(d) + b_h(dx(p,q), dy(p,q))
```

The bias tables are initialized to zero. The first forward therefore starts
from the V1 content-attention function and learns a position-dependent
correction.

The snapshot deliberately reuses only the maintained common trainer for data
loading, the frozen teacher, DeiT construction, optimizer, scheduler,
evaluation, and atomic checkpoint I/O. It does not import any model or loss
implementation from `methods/OursV2`. This mirrors the established
`serve/table4_grid_permutation` review format while avoiding a divergent copy
of 1,400 lines of generic training infrastructure.

## Review/run command

```bash
python serve/ours_v2/train_cifar100_full.py \
  --num-workers 4 \
  --output-dir /app/output \
  --run-name ours_v2_relative_position_v1_cifar100_300ep_seed1
```

The maintained experimental entry point remains:

```bash
python methods/OursV2/cifar100/train.py --num-workers 4
```

Regression tests verify that the consolidated and maintained models have
identical state dictionaries, forward outputs, losses, and protocol defaults.
