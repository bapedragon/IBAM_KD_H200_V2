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

`train_pure_alg.py` is the method-isolated ALG-paper/public-LG entry point.
Unlike researcher-sync `train.py`, it follows the published ALG derivative
equations, uses `derivative >= -0.02`, and has no extra controller warm-up.
The optimizer still has the published 20-epoch LR warm-up. Batch size is an
explicit ablation axis, so batch 128 and 64 receive separate protocol and run
names:

```bash
python methods/ALG/chaoyang/train_pure_alg.py --batch-size 128 \
  --protocol-name chaoyang_deit_ti_alg_paper_lg_v2_b128 --timing-run

python methods/ALG/chaoyang/train_pure_alg.py --batch-size 64 \
  --protocol-name chaoyang_deit_ti_alg_paper_lg_v2_b64 --timing-run
```

```bash
python methods/ALG/chaoyang/train_draft_common.py --timing-run --num-workers 4
python methods/ALG/chaoyang/train_draft_common.py --output-dir /app/output \
  --run-name alg_chaoyang_deit_ti_draft_common_300ep_seed42 --num-workers 4
```
