# Table 7 control: loss-balance sensitivity

This isolated folder fills the CIFAR-100/DeiT-Ti loss-balance sweep in Table
7. Its full-Ours reference is the verified researcher-synchronized run at
`lambda=0.5`, `82.90%` Top-1. The primary Ours implementation and its result
are not modified.

## One-factor contract

Every run inherits the complete protocol from
[`../cifar100/train.py`](../cifar100/train.py). Only `lambda` changes in:

```text
L_total(e) = CE + beta(e) * [lambda * L_fuse + (1-lambda) * L_align]
```

| Lambda | Weighted feature loss | Table status |
|---:|---|---|
| `0` | `L_align` | complete: `83.29%` |
| `0.25` | `0.25 L_fuse + 0.75 L_align` | complete: `83.40%` |
| `0.5` | `0.5 L_fuse + 0.5 L_align` | reuse verified `82.90%` |
| `0.75` | `0.75 L_fuse + 0.25 L_align` | new run |
| `1.0` | `L_fuse` | new run |

The independent-weight `(lambda_1, lambda_2)` column is deliberately not
implemented yet. V3 requests one such row but does not state the pair to use.
It must be fixed and documented before adding a separate independent-weight
entry point; it must not be silently conflated with the convex lambda sweep.

## Locked protocol

| Item | Value |
|---|---|
| Dataset | CIFAR-100, train 50,000 / test 10,000 |
| Teacher | fixed V2 ResNet56, 32 px, selected Top-1 `71.91%` |
| Student | DeiT-Ti from scratch, 224 px |
| Epochs | 300 |
| Train / eval batch | 64 / 200 |
| Optimizer | AdamW, LR `5e-4`, min LR `5e-6`, weight decay `0.05` |
| LR schedule | 20-epoch warm-up, factor `0.001`, then cosine |
| Regularization | label smoothing `0`, drop path `0.1` |
| Augmentation | researcher-sync/public-LG strong augmentation |
| Runtime | FP32, seed `1`, workers `4` |
| Grid | larger-grid policy: `32/16/14` |
| Adaptive guidance | beta `2.5`, tau `-0.02`, window `50`, gate warm-up `20` |
| Selection/reporting | best CIFAR-100 test Top-1 |

The adaptive-controller implementation is unchanged. For each lambda run it
observes that run's weighted feature loss shown in the table above.

## H200 commands

Run one full-dataset, two-epoch timing check first:

```bash
python methods/Ours/table7_loss_balance/train_cifar100.py \
  --lambda-value 0 --timing-run --num-workers 4 \
  --run-name table7_lambda_0_cifar100_timing_2ep
```

After the audit passes, run the corresponding 300-epoch experiment:

```bash
python methods/Ours/table7_loss_balance/train_cifar100.py \
  --lambda-value 0 --student-epochs 300 --num-workers 4 \
  --run-name table7_lambda_0_cifar100_300ep_seed1 \
  --output-dir /app/output
```

Replace only the value and run name for `0.25`, `0.75`, or `1.0`. Do not pass
`--fusion-ratio` directly; the wrapper rejects it to keep the audit trail
unambiguous. The final `ours_best_top1` fills the matching Table 7 cell.

## Expected audit lines

For the first `lambda=0` timing run, startup must contain:

```text
[TABLE7_CONTROL] reference_lambda=0.5 reference_full_top1=82.90% only_change=lambda lambda=0
[TABLE7_LOSS] feature_loss=0*L_fuse+1*L_align ...
[OURS] ... lambda=0.0
```

If any other protocol field differs from the normal CIFAR-100 Ours run, do
not accept the result as a Table 7 control.

## Completed results

| Lambda | Best epoch | Best Top-1 | Last Top-1 | Full-Ours gap | Guidance stop |
|---:|---:|---:|---:|---:|---:|
| 0 | 269 | **83.29%** | 83.17% | +0.39 pp | 119 |
| 0.25 | 289 | **83.40%** | 83.24% | +0.50 pp | 117 |
| 0.5 | 288 | **82.90%** | 82.62% | reference | 117 |

The first two checkpoints and summaries are imported under
`results/Ours/cifar100/table7_lambda_0_researcher_sync_v1_300ep_seed1/` and
`results/Ours/cifar100/table7_lambda_0p25_researcher_sync_v1_300ep_seed1/`.
Values for `0.75`, `1.0`, and the independent pair remain unfilled until their
own complete artifacts are received.
