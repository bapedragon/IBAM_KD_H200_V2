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

Account names and H200 build numbers are kept only under `run_logs`; they do
not determine checkpoint placement. `PENDING_IMPORTS.md` records jobs that
have started but whose output archives have not yet been verified and added.

Only the selected best checkpoint is committed. The original downloaded
outputs retain `student_latest.pt`; its exact final-epoch accuracy is also
preserved in `run_summary.json`. This avoids doubling repository size and H200
clone time without discarding the reported result.

All 33 currently committed checkpoints were loaded with PyTorch and verified against
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
| ALG | Scheduled match (static) |  |  |  |
| **Ours** | **Grid-space, learnable** | **82.90** | **74.81** | **81.95\*** |

Blank cells mean not yet run under the intended method-specific protocol; they
do not mean zero accuracy. The table intentionally excludes historical/mixed ALG
diagnostics. `*` means the Chaoyang Ours result is verified from the completed
H200 log, while its checkpoint and summary archive are still pending import.
The Vanilla values are draft references; every other populated cell is a
reproduction result from this project.

Protocol families used in this table:

- generic CIFAR-100: `generic_kd_v2_300ep_seed42`;
- generic Flowers/Chaoyang: `generic_kd_300ep_epoch_only_v1_seed42`;
- Ours: `researcher_sync_v1_300ep_seed1`.

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

Researcher-sync runs using train+val (`2,040`) / test (`6,149`):

| Method | Best epoch | Best Top-1 | Last Top-1 | Working-paper value | Gap |
|---|---:|---:|---:|---:|---:|
| ALG | 288 | **75.02%** | 74.87% | 68.54% | +6.48 pp |
| Ours | 251 | **74.81%** | 74.21% | 70.31% | +4.50 pp |

These researcher-sync Flowers results are recorded by exact protocol ID. They
must not be confused with the later method-separated ALG/Ours entry points or
with the three-way split audit below.

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

Generic methods use the CNN-to-ViT adapters documented in each method
directory. They should not be described as unmodified original CNN-to-CNN
experiments. Ours is maintained separately from the five generic baselines.
