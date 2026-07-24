# Canonical ALG: CIFAR-100 / DeiT-Ti

`train.py` uses the official LG base plus the published ALG controller. It
locks batch `128`, optimizer LR warm-up `20`, controller warm-up `0`, paper
derivative equations, `>= -0.02` stopping, FP32, and seed `1`.

```bash
python methods/ALG/cifar100/train.py --timing-run --num-workers 4
python methods/ALG/cifar100/train.py --num-workers 4 \
  --run-name alg_cifar100_deit_ti_300ep_seed1 --output-dir /app/output
```

The teacher consumes the bilinearly resized 32-pixel view. The student uses
the official LG 224-pixel augmentation and feature path.
