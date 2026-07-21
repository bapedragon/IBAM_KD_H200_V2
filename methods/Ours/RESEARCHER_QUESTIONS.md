# Researcher confirmation checklist for Ours/ALG

This checklist separates what is visible in the supplied screenshots/source
from what is still unknown. The active implementation is reproducible, but it
must be called `researcher_sync_v1`, not a byte-identical original run, until
the open items are answered.

## Confirmed from the supplied configuration or code

| Item | Confirmed value |
|---|---|
| Student | DeiT-Ti: patch 16, dim 192, depth 12, 3 heads, MLP ratio 4 |
| Student / teacher resolution | 224 / 32 |
| Teacher / student feature indexes | `[0,1,2]` / `[0,6,11]` for ALG/LG |
| Feature transform / weight | linear (`1x1`) / `2.5` |
| Ours feature resize | both teacher and student to the larger grid, bilinear |
| Ours guidance loss | sum of `0.5*alignment + 0.5*fusion` over stages |
| Trainer objective | CE plus beta times guidance while guidance is active |
| ALG controller | beta `2.5`, threshold `-0.02`, window 50, warm-up 20 |
| Student optimizer | AdamW, LR `5e-4`, minimum LR `5e-6`, cosine, WD `0.05` |
| Epochs / optimizer warm-up | 300 / 20 |
| Train / test batch in shown config | 64 / 200 |
| Drop path | `0.1` |

## Ask the researcher

| Priority | Question | Current reproducible choice | Why it matters |
|---:|---|---|---|
| 1 | Is `cur_epoch` passed to `update_epoch()` as zero-based or one-based? Please share the exact call line. | one-based adapter | can shift the controller stop decision by one epoch |
| 1 | Is the reported Top-1/checkpoint selected from the normal model or EMA model? | normal model best Top-1 | screenshots show an EMA path, but not the reporting/selection rule |
| 1 | Were reported runs trained with online teacher inference or precomputed `offline_features`? | online frozen teacher | changes the exact teacher feature tensors and data path |
| 1 | Was logit distillation enabled in addition to feature guidance? | disabled | changes the total objective directly |
| 1 | Does train batch 64 mean global batch 64 across all GPUs? | global 64 on one H200 | affects optimization and comparison with the original multi-GPU run |
| 2 | What is the exact `inter_distill_loss()` definition and reduction (`mean`, `sum`, or other)? | elementwise MSE mean per stage, summed across stages | changes feature-loss scale |
| 2 | Which teacher checkpoint was used for each dataset, including its Top-1 and hash? | repository's fixed selected teacher per dataset | teacher quality/features affect every method |
| 2 | What exact augmentation, normalization, mixup, CutMix, label-smoothing, and random-erasing values were active? | public LG defaults shown in the run log | screenshots do not expose the entire data config |
| 2 | What random seed, pycls/tiny-transformers commit, PyTorch version, and timm version produced the table values? | seed 1, audited LG commit, current H200 packages | needed for a stronger reproduction claim |

## Change-control rule

An answer must be recorded with its source (message, code line, config, or
commit). If it changes training behavior, create a new protocol ID such as
`researcher_sync_v2`; never overwrite `researcher_sync_v1` artifacts or rename
historical results as though they used the new setting.
