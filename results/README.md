# Curated student results

This directory is the compact, reporting-oriented view of completed H200
student runs. Raw Pod output folders contained repeated batch wrappers and both
`best` and `latest` checkpoints. Here they are normalized to:

```text
results/
├── <Method>/<dataset>/<protocol-id>/
│   ├── run_summary.json
│   └── student_best.pt
├── PENDING_IMPORTS.md
├── run_logs/
└── CHECKSUMS.sha256
```

The protocol-ID directory is mandatory. No checkpoint or summary may be placed
directly under a dataset directory. This prevents a new researcher-sync run
from overwriting or being mistaken for an older run of the same method and
dataset. The canonical IDs currently used are:

| Protocol ID | Meaning |
|---|---|
| `generic_kd_v2_300ep_seed42` | completed CIFAR-100 generic KD-family run |
| `generic_kd_v2_200ep_seed42_historical` | historical Flowers generic run |
| `generic_kd_v2_100ep_seed42_historical` | historical Chaoyang generic run |
| `generic_kd_300ep_epoch_only_v1_seed42` | Flowers/Chaoyang generic rerun with the earlier recipe and a 300-epoch horizon |
| `pre_researcher_sourcegrid_300ep_seed42_historical` | pre-sync Ours CIFAR run |
| `pre_researcher_papergrid_100ep_seed42_historical` | pre-sync Ours Chaoyang run |
| `pre_researcher_batch128_300ep_seed1_historical` | pre-sync ALG Chaoyang run |
| `researcher_sync_v1_300ep_seed1` | researcher-synchronized Ours/ALG family; Flowers uses train+val 2,040 / test 6,149 |
| `researcher_sync_v2_official_three_way_300ep_seed1_historical` | historical Flowers run: train 1,020, val-best 1,020, final test 6,149 once |
| `table4_grid_permuted_researcher_sync_v1_300ep_seed1_permseed1` | Table 4 global, fixed, stage-wise spatial permutation; the same permutation feeds K/V and both feature targets |
| `table7_lambda_0_researcher_sync_v1_300ep_seed1` | Table 7 alignment-only endpoint (`lambda=0`) |
| `table7_lambda_0p25_researcher_sync_v1_300ep_seed1` | Table 7 `0.25 L_fuse + 0.75 L_align` |
| `paper_lg_v2_trainval_test_b128_300ep_seed1` | selected pure-ALG Flowers train batch 128 |
| `paper_lg_v2_trainval_test_b64_300ep_seed1` | pure-ALG Flowers train batch 64 control |
| `paper_source_v2_trainval_test_b128_300ep_seed1` | Ours Flowers batch-128 protocol-separated control |
| `paper_lg_v2_b128_300ep_seed1` | pure-ALG Chaoyang batch-128 control |
| `paper_lg_v2_b64_300ep_seed1` | selected pure-ALG Chaoyang batch-64 result |
| `cifar100_locked_b64_v1_300ep_seed1` | Ours Chaoyang CIFAR-100-locked batch-64 control |

Account names and H200 build numbers are kept only under `run_logs`; they do
not determine checkpoint placement. `PENDING_IMPORTS.md` records jobs that
have started but whose output archives have not yet been verified and added.

Only the selected best checkpoint is committed. The original downloaded
outputs retain `student_latest.pt`; its exact final-epoch accuracy is also
preserved in `run_summary.json`. This avoids doubling repository size and H200
clone time without discarding the reported result.

All 42 currently committed checkpoints were loaded with PyTorch and verified against
their summaries for dataset, method, best accuracy, and checkpoint epoch.
The Top-1 value is read from the adjacent summary; file names are deliberately
stable (`student_best.pt`) inside the provenance-rich protocol directory.

## Consolidated DeiT-Ti reproduction table

| Method | Transfer operator | CIFAR-100 | Flowers-102 | Chaoyang |
|---|---|---:|---:|---:|
| Vanilla DeiT-Ti | - | 65.08 | 50.06 | 82.00 |
| KD | Logits | 69.10 | 48.95 | 62.79 |
| CRD | Pooled contrastive | 68.59 | 49.06 | 79.85 |
| ReviewKD | Projected fusion | 75.65 | 61.88 | 82.75 |
| MGD | Masked reconstruction | 75.68 | 54.66 | 81.81 |
| OFA | Logit-space projection | 67.73 | 46.41 | 78.03 |
| LG | Direct match (static) |  |  |  |
| ALG | Scheduled match (static) |  | 73.15 | 83.54 |
| **Ours** | **Grid-space, learnable** | **82.90** | **74.81** | **81.95\*** |

Blank cells mean not yet run under the intended method-specific protocol; they
do not mean zero accuracy. Flowers ALG uses train batch 128 (`73.15%`) and
Chaoyang ALG uses train batch 64 (`83.54%`); both checkpoints and summaries
are verified. `*` marks the pending-artifact status for the Chaoyang Ours
result. The Vanilla values are draft references; every other populated cell
is a reproduction result from this project.

Protocol families used in this table:

- generic CIFAR-100: `generic_kd_v2_300ep_seed42`;
- generic Flowers/Chaoyang: `generic_kd_300ep_epoch_only_v1_seed42`;
- Flowers ALG: `paper_lg_v2_trainval_test_b128_300ep_seed1`;
- Chaoyang ALG: `paper_lg_v2_b64_300ep_seed1`;
- Ours: `researcher_sync_v1_300ep_seed1` (Flowers train batch 64).

## CIFAR-100

Shared setup: ResNet56 teacher at 32 x 32 and scratch DeiT-Ti student at
224 x 224 for 300 epochs. Generic methods use seed 42; the researcher-sync
Ours run uses seed 1.

| Method | Best epoch | Best Top-1 | Last Top-1 | Vanilla gap | Status |
|---|---:|---:|---:|---:|---|
| Vanilla DeiT-Ti | - | 65.08% | - | - | Draft reference |
| KD | 191 | **69.10%** | 68.59% | +4.02 pp | Verified |
| CRD | 79 | **68.59%** | 66.74% | +3.51 pp | Verified |
| Ours (researcher sync) | 288 | **82.90%** | 82.62% | +17.82 pp | Verified; current synchronized run |
| Ours (pre-sync) | 296 | **79.52%** | 79.49% | +14.44 pp | Historical source-grid run |
| ReviewKD | 233 | **75.65%** | 75.50% | +10.57 pp | Verified |
| MGD | 215 | **75.68%** | 75.31% | +10.60 pp | Verified |
| OFA | 263 | **67.73%** | 67.50% | +2.65 pp | Verified |

The researcher-sync Ours result is `82.90%`. The working-paper value recorded
by the repository is `82.42%`, a difference of `+0.48 pp` (the separately
communicated `82.43%` value gives `+0.47 pp`). The older `79.52%` checkpoint
remains in a distinct historical protocol directory and was not overwritten.

### Table 4 attribution control

The completed global grid-permutation control changes only the teacher spatial
ordering. It uses one fixed permutation per stage and applies that same
permutation to K, V, the alignment target, and the fusion target.

| Configuration | Best epoch | Best Top-1 | Last Top-1 | Gap to full Ours | Artifact status |
|---|---:|---:|---:|---:|---|
| Full Ours | 288 | **82.90%** | 82.62% | - | Verified |
| Grid-permuted teacher features | 298 | **81.79%** | 81.61% | -1.11 pp | Verified |

The measured `81.79%` is `+1.99 pp` above the draft token-space row
(`79.80%`). It therefore does not support a claimed collapse below that row.
The result must be reported as measured or followed by a separately defined
permutation intervention; it must not be replaced with an expected value.
Artifacts:
`Ours/cifar100/table4_grid_permuted_researcher_sync_v1_300ep_seed1_permseed1/`.

### Table 7 loss-balance controls

All rows below reuse the exact full-Ours CIFAR-100 protocol and change only the
convex feature-loss balance:
`lambda L_fuse + (1-lambda) L_align`.

| Lambda | Feature loss | Best epoch | Best Top-1 | Last Top-1 | Gap to full Ours | Status |
|---:|---|---:|---:|---:|---:|---|
| 0 | `L_align` | 269 | **83.29%** | 83.17% | +0.39 pp | Verified |
| 0.25 | `0.25 L_fuse + 0.75 L_align` | 289 | **83.40%** | 83.24% | +0.50 pp | Verified |
| 0.5 | `0.5 L_fuse + 0.5 L_align` | 288 | **82.90%** | 82.62% | reference | Reused verified full Ours |
| 0.75 | `0.75 L_fuse + 0.25 L_align` | - | - | - | - | Awaiting result |
| 1.0 | `L_fuse` | - | - | - | - | - | Awaiting result |
| `(lambda_1, lambda_2)` | independent weights | - | - | - | - | Pair not yet fixed |

The imported endpoints are stored at
`Ours/cifar100/table7_lambda_0_researcher_sync_v1_300ep_seed1/` and
`Ours/cifar100/table7_lambda_0p25_researcher_sync_v1_300ep_seed1/`.

## Flowers-102 — completed 300-epoch results

The five generic methods below retain their earlier Flowers hyperparameters
(batch 64, warm-up 5, seed 42) and change only the epoch/cosine horizon from
200 to 300. Training uses the official train+val images (`2,040`) and reports
on the official test set (`6,149`).

| Method | Best epoch | Best Top-1 | Last Top-1 | Vanilla gap |
|---|---:|---:|---:|---:|
| Vanilla DeiT-Ti | - | 50.06% | - | - |
| KD | 105 | **48.95%** | 46.69% | -1.11 pp |
| CRD | 172 | **49.06%** | 48.06% | -1.00 pp |
| ReviewKD | 256 | **61.88%** | 61.52% | +11.82 pp |
| MGD | 248 | **54.66%** | 54.09% | +4.60 pp |
| OFA | 201 | **46.41%** | 45.54% | -3.65 pp |

### Selected method-specific Flowers results

Both selected rows use train+val (`2,040`) / test (`6,149`), 300 epochs, seed
1, and a scratch DeiT-Ti at 224 x 224. They intentionally preserve their own
method protocols instead of forcing one method's batch setting onto the other.

| Method | Train / eval batch | Best epoch | Best Top-1 | Last Top-1 | Working-paper value | Gap | Artifact status |
|---|---:|---:|---:|---:|---:|---:|---|
| ALG (paper/public-LG protocol) | 128 / 200 | 274 | **73.15%** | 72.92% | 68.54% | +4.61 pp | Checkpoint and summary verified |
| Ours (CIFAR-100-matched researcher sync) | 64 / 200 | 251 | **74.81%** | 74.21% | 70.31% | +4.50 pp | Checkpoint and summary verified |

The Ours batch-64 artifacts are committed at
`Ours/flowers102/researcher_sync_v1_300ep_seed1/`. The ALG batch-128 artifacts
are committed at `ALG/flowers102/paper_lg_v2_trainval_test_b128_300ep_seed1/`.
The same sequence produced Ours batch-128 best `72.78%` at epoch 264, last
`72.29%`, and guidance-stop epoch 211; its artifacts are kept at
`Ours/flowers102/paper_source_v2_trainval_test_b128_300ep_seed1/` as an
auxiliary protocol result rather than the selected batch-64 Ours row.

### Auxiliary researcher-sync comparison

The earlier shared researcher-sync batch used train/eval batch `64/200` for
both methods. It remains useful as an ablation, but its ALG row is not the
selected pure ALG-paper result in the consolidated table.

| Method | Best epoch | Best Top-1 | Last Top-1 | Working-paper value | Gap |
|---|---:|---:|---:|---:|---:|
| ALG (researcher-sync batch 64) | 288 | **75.02%** | 74.87% | 68.54% | +6.48 pp |
| Ours (researcher-sync batch 64) | 251 | **74.81%** | 74.21% | 70.31% | +4.50 pp |

The pure-ALG batch comparison additionally produced Flowers batch-64
`75.05%` (epoch 298, last `74.60%`, guidance stop 185). The later
method-separated batch-128 sequence produced Ours `72.78%` and ALG `73.15%`;
only ALG batch 128 is selected from that sequence because the requested Ours
reporting row is the CIFAR-100-matched batch-64 run. These results must not be
confused with the three-way split audit below.

### Historical official three-way split audit

The `researcher_sync_v2_official_three_way_300ep_seed1_historical` runs used
train `1,020`, validation `1,020`, and test `6,149`. The checkpoint was selected
on validation and the test set was evaluated once.

| Method | Best val epoch | Best val Top-1 | Final test Top-1 |
|---|---:|---:|---:|
| ALG | 269 | 71.57% | **63.57%** |
| Ours | 288 | 70.10% | **61.10%** |

### Historical 200-epoch generic results

Fixed protocol: ResNet56 teacher at 32 x 32, scratch DeiT-Ti student at
224 x 224, 200 epochs, seed 42.

| Method | Best epoch | Best Top-1 | Last Top-1 | Vanilla gap |
|---|---:|---:|---:|---:|
| Vanilla DeiT-Ti | - | 50.06% | - | - |
| KD | 105 | **47.91%** | 46.77% | -2.15 pp |
| CRD | 91 | **49.49%** | 48.20% | -0.57 pp |
| ReviewKD | 149 | **58.89%** | 58.72% | +8.83 pp |
| MGD | 172 | **53.42%** | 53.21% | +3.36 pp |
| OFA | 159 | **46.09%** | 45.55% | -3.97 pp |

## Chaoyang — completed 300-epoch generic results

The five methods retain the earlier Chaoyang recipe (batch 64, warm-up 5,
seed 42) and change only the epoch/cosine horizon from 100 to 300.

| Method | Best epoch | Best Top-1 | Last Top-1 | Vanilla gap |
|---|---:|---:|---:|---:|
| Vanilla DeiT-Ti | - | 82.00% | - | - |
| KD | 49 | **62.79%** | 57.60% | -19.21 pp |
| CRD | 189 | **79.85%** | 78.45% | -2.15 pp |
| ReviewKD | 166 | **82.75%** | 81.25% | +0.75 pp |
| MGD | 155 | **81.81%** | 80.93% | -0.19 pp |
| OFA | 212 | **78.03%** | 75.88% | -3.97 pp |

### Historical 100-epoch generic results

All rows use the fixed ResNet56 teacher at 32 x 32 and a scratch DeiT-Ti
student at 224 x 224. The five generic methods and the historical Ours run use
100 epochs and seed 42. The stored ALG row below is the earlier pre-sync
300-epoch, batch-128 result. Any researcher-synchronized Chaoyang ALG result
must be imported under a new protocol ID rather than overwriting this row.

| Method | Best epoch | Best Top-1 | Last Top-1 | Vanilla gap | Status |
|---|---:|---:|---:|---:|---|
| Vanilla DeiT-Ti | - | 82.00% | - | - | Draft reference |
| ALG | 235 | **80.32%** | 79.71% | -1.68 pp | Verified pre-sync batch-128 run |
| Ours | 82 | **81.21%** | 80.46% | -0.79 pp | Historical 100-epoch paper-grid run |
| KD | 15 | **62.79%** | 56.80% | -19.21 pp | Verified |
| CRD | 61 | **79.66%** | 77.93% | -2.34 pp | Verified |
| ReviewKD | 86 | **81.72%** | 81.07% | -0.28 pp | Verified |
| MGD | 80 | **80.69%** | 79.94% | -1.31 pp | Verified |
| OFA | 90 | **75.55%** | 74.99% | -6.45 pp | Verified |

The stored Ours result is the earlier 100-epoch, seed-42 run with teacher-grid
targets `32 x 32`, `16 x 16`, and `8 x 8`. Its checkpoint and summary are
explicitly named `historical` so they cannot be confused with the pending
300-epoch matched ALG-base reruns.

### Pure-ALG batch comparison and auxiliary Ours run

Build 480 completed four 300-epoch, seed-1 runs and preserved the full log,
sequence summary, and individual checkpoints/per-run summaries.

| Method | Train batch | Best epoch | Best Top-1 | Last Top-1 | Guidance stop |
|---|---:|---:|---:|---:|---:|
| ALG | 128 | 281 | **80.97%** | 80.46% | 213 |
| ALG | 64 | 292 | **83.54%** | 82.84% | 183 |
| Ours (CIFAR-100-locked protocol) | 64 | 271 | **81.11%** | 80.22% | 192 |

The ALG batch-64 result is only `+0.04 pp` from the working-paper `83.50%`
value and is the intended ALG Chaoyang reproduction shown in the consolidated
table. The Ours `81.11%` row is an auxiliary CIFAR-100-locked run and does not
replace the separate pending `81.95%` researcher-sync Ours result. The three
ALG artifacts are under `ALG/flowers102/paper_lg_v2_trainval_test_b64_300ep_seed1/`,
`ALG/chaoyang/paper_lg_v2_b128_300ep_seed1/`, and
`ALG/chaoyang/paper_lg_v2_b64_300ep_seed1/`; the Ours artifact is under
`Ours/chaoyang/cifar100_locked_b64_v1_300ep_seed1/`.

## Import status

The 300-epoch generic Flowers/Chaoyang batch and the researcher-sync Ours/ALG
batch have been imported and verified. Historical 200/100-epoch results remain
in separate protocol directories; no old checkpoint was overwritten.

## Source runs

- `run_logs/h200_build-450_combined-generic-kd.log`: Chaoyang five methods,
  Flowers-102 five methods, then CIFAR-100 KD.
- `run_logs/h200_build-452_cifar100-ours-crd.log`: CIFAR-100 Ours and CRD.
- `run_logs/h200_build-453_cifar100-reviewkd-mgd.log`: CIFAR-100 ReviewKD
  and MGD.
- `run_logs/h200_build-454_cifar100-ofa.log`: CIFAR-100 OFA.
- `run_logs/h200_build-457_chaoyang-ours-papergrid-100ep.log`: historical
  Chaoyang Ours paper-grid run (100 epochs, seed 42).
- `run_logs/h200_build-461_chaoyang-alg-public-base-300ep.log`: Chaoyang ALG
  run on the audited public LG/ALG base (300 epochs, seed 1).
- `run_logs/h200_build-471_generic-kd-flowers-chaoyang-300ep.log`: the ten
  completed 300-epoch generic Flowers/Chaoyang runs.
- `run_logs/h200_build-475_researcher-sync-ours-alg-300ep.log`: Ours
  CIFAR-100, Ours Flowers-102, and ALG Flowers-102 researcher-sync runs.
- `run_logs/h200_build-477_flowers-official-three-way-ours-alg-300ep.log`:
  historical Flowers train/val/test audit.
- `run_logs/h200_build-479_flowers-alg-ours-protocol-separated-300ep.log`:
  pure ALG batch-128 and Ours batch-128 Flowers sequence; adjacent sequence
  JSON records task commands and best values.
- `run_logs/h200_build-480_alg-batch-comparison-ours-chaoyang-300ep.log`:
  ALG Flowers batch 64, ALG Chaoyang batches 128/64, and Ours Chaoyang batch
  64; adjacent sequence JSON records the four completed tasks.
- `run_logs/h200_build-482_table4-grid-permutation-cifar100-300ep.log`:
  Table 4 grid-permutation control (`81.79%`).
- `run_logs/h200_build-484_table7-lambda0-cifar100-300ep.log`:
  Table 7 `lambda=0` control (`83.29%`).
- `run_logs/h200_build-485_table7-lambda0p25-cifar100-300ep.log`:
  Table 7 `lambda=0.25` control (`83.40%`).
- `run_logs/flowers102_alg128_ours128_300ep_final_excerpt.txt`: supplied final
  lines from the method-separated Flowers sequence (ALG batch 128 `73.15%`,
  Ours batch 128 `72.78%`); superseded for auditing by the complete build-479
  log but retained as the originally supplied excerpt.

Generic methods use the CNN-to-ViT adapters documented in each method
directory. They should not be described as unmodified original CNN-to-CNN
experiments. Ours is maintained separately from the five generic baselines.
