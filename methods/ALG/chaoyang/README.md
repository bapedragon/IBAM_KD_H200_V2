# ALG / Chaoyang / DeiT-Ti

Independent Chaoyang entry point for the protocol in
[`../README.md`](../README.md).

```bash
python methods/ALG/chaoyang/train.py --timing-run --num-workers 4
python methods/ALG/chaoyang/train.py --output-dir /app/output \
  --run-name alg_chaoyang_researcher_sync_300ep_seed1 --num-workers 4
```

Fixed comparison target: ALG paper Table II `83.50%` Top-1, with the CNN
guidance ending at epoch `108`. The official Chaoyang train/test splits are
`4,021/2,139` images.

## Two deliberately separate ALG protocol families

`train.py` is the researcher-synchronized LG/ALG-base run used with the recent
ALG/Ours comparison. It fixes batch 64 and uses the same exact adaptive
controller as the synchronized Ours run: controller warm-up 20, window 50,
strict `smoothed_derivative > -0.02`, and no descent-first guard.
`train_draft_common.py` remains a historical diagnostic with its original
batch 128 training base and must not be mixed into the synchronized table.

| Family | Entry point | Seed | Augmentation / evaluation | Regularization |
|---|---|---:|---|---|
| Researcher-synchronized LG/ALG base | `train.py` | 1 | strong LG / direct 224 resize | batch 64, label smoothing 0, drop path 0.1, FP32 |
| Draft-common historical base | `train_draft_common.py` | 42 | RRC+flip / Resize256+CenterCrop224 | label smoothing 0.1, drop path 0, AMP |

The second run is a controlled protocol diagnostic. Results must be compared
within a family, not presented as a repeated seed of the first family.

```bash
python methods/ALG/chaoyang/train_draft_common.py --timing-run --num-workers 4
python methods/ALG/chaoyang/train_draft_common.py --output-dir /app/output \
  --run-name alg_chaoyang_deit_ti_draft_common_300ep_seed42 --num-workers 4
```
