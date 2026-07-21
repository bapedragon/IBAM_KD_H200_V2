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

## Two deliberately separate ALG protocol families

`train.py` is the audited public LG/ALG-base run used with the recent
ALG/Ours comparison. `train_draft_common.py` changes only the shared training
base to the historical configuration used by the earlier Ours `81.11%` run;
the ALG operator, beta controller, teacher checkpoint, dataset split, 300
epochs, batch 128, warm-up 20, and 224/32 image geometry remain fixed.

| Family | Entry point | Seed | Augmentation / evaluation | Regularization |
|---|---|---:|---|---|
| Public LG/ALG base | `train.py` | 1 | strong LG / direct 224 resize | label smoothing 0, drop path 0.1, FP32 |
| Draft-common historical base | `train_draft_common.py` | 42 | RRC+flip / Resize256+CenterCrop224 | label smoothing 0.1, drop path 0, AMP |

The second run is a controlled protocol diagnostic. Results must be compared
within a family, not presented as a repeated seed of the first family.

```bash
python methods/ALG/chaoyang/train_draft_common.py --timing-run --num-workers 4
python methods/ALG/chaoyang/train_draft_common.py --output-dir /app/output \
  --run-name alg_chaoyang_deit_ti_draft_common_300ep_seed42 --num-workers 4
```
