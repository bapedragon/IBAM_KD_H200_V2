# ALG / Chaoyang / DeiT-Ti

Independent Chaoyang entry point for the protocol in
[`../README.md`](../README.md).

```bash
python methods/ALG/chaoyang/train.py --timing-run --num-workers 4
python methods/ALG/chaoyang/train.py --output-dir /app/output \
  --run-name alg_chaoyang_deit_ti_300ep_seed1 --num-workers 4
```

Fixed comparison target: ALG paper Table II `83.50%` Top-1, with the CNN
guidance ending at epoch `108`. The official Chaoyang train/test splits are
`4,021/2,139` images.
