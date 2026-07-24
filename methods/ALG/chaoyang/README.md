# Canonical ALG: Chaoyang / DeiT-Ti

The active `train.py` is the ALG-paper controller on the official LG base:
batch `128`, FP32, seed `1`, 20-epoch optimizer LR warm-up, no controller-only
warm-up, paper derivative normalization, and the `>= -0.02` stop boundary.

```bash
python methods/ALG/chaoyang/train.py --timing-run --num-workers 4
python methods/ALG/chaoyang/train.py --output-dir /app/output \
  --run-name alg_chaoyang_300ep_seed1 --num-workers 4
```

The paper comparison target is `83.50%` Top-1 with guidance ending at epoch
`108`. The official Chaoyang split contains `4,021/2,139` train/test images.

Completed batch-128 and batch-64 historical paper-controller results remain
under `results/ALG/chaoyang/` as ablations. The former `draft_common` and
researcher-sync entry points are archived under `methods/ALG/legacy` and are
not valid canonical ALG runs.
