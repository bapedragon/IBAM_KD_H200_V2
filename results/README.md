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
| `pre_researcher_sourcegrid_300ep_seed42_historical` | pre-sync Ours CIFAR run |
| `pre_researcher_papergrid_100ep_seed42_historical` | pre-sync Ours Chaoyang run |
| `pre_researcher_batch128_300ep_seed1_historical` | pre-sync ALG Chaoyang run |
| `researcher_sync_v1_300ep_seed1` | current researcher-synchronized Ours/ALG family |
| `generic_kd_300ep_epoch_only_v1_seed42` | pending 300-epoch Flowers/Chaoyang generic reruns |

Account names and H200 build numbers are kept only under `run_logs`; they do
not determine checkpoint placement. `PENDING_IMPORTS.md` records jobs that
have started but whose output archives have not yet been verified and added.

Only the selected best checkpoint is committed. The original downloaded
outputs retain `student_latest.pt`; its exact final-epoch accuracy is also
preserved in `run_summary.json`. This avoids doubling repository size and H200
clone time without discarding the reported result.

All 18 currently committed checkpoints were loaded with PyTorch and verified against
their summaries for dataset, method, best accuracy, and checkpoint epoch.
The Top-1 value is read from the adjacent summary; file names are deliberately
stable (`student_best.pt`) inside the provenance-rich protocol directory.

## CIFAR-100

Fixed protocol: ResNet56 teacher at 32 x 32, scratch DeiT-Ti student at
224 x 224, 300 epochs, seed 42.

| Method | Best epoch | Best Top-1 | Last Top-1 | Vanilla gap | Status |
|---|---:|---:|---:|---:|---|
| Vanilla DeiT-Ti | - | 65.08% | - | - | Draft reference |
| KD | 191 | **69.10%** | 68.59% | +4.02 pp | Verified |
| CRD | 79 | **68.59%** | 66.74% | +3.51 pp | Verified |
| Ours | 296 | **79.52%** | 79.49% | +14.44 pp | Historical source-grid run; not the paper-grid table result |
| ReviewKD | 233 | **75.65%** | 75.50% | +10.57 pp | Verified |
| MGD | 215 | **75.68%** | 75.31% | +10.60 pp | Verified |
| OFA | 263 | **67.73%** | 67.50% | +2.65 pp | Verified |

The current Ours checkpoint used the supplied larger-grid rule, producing
stage targets `32 x 32`, `16 x 16`, and `14 x 14`. The experiment policy is
now fixed to V3's teacher-resolution rule, which produces `32 x 32`,
`16 x 16`, and `8 x 8` with the committed ResNet56. Therefore this historical
checkpoint remains reproducible but must not be reported as the final
paper-grid result; CIFAR-100 requires a separately labeled rerun.

## Flowers-102 — historical 200-epoch results

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

## Chaoyang — historical 100-epoch generic results

All rows use the fixed ResNet56 teacher at 32 x 32 and a scratch DeiT-Ti
student at 224 x 224. The five generic methods and the historical Ours run use
100 epochs and seed 42. The stored ALG row below is the earlier pre-sync
300-epoch, batch-128 result. The current researcher-synchronized ALG entry
point instead fixes batch 64 and the exact supplied controller; its result is
pending and must be recorded as a new run rather than overwriting provenance.

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

## Current 300-epoch reruns

The old checkpoints above remain immutable historical records. New runs use
separate collection roots and provenance-rich names:

| Run family | Output collection root | Seed / base |
|---|---|---|
| Generic Flowers + Chaoyang | `/app/output/generic_kd_flowers_chaoyang_300ep_seed42` | 42 / previous generic protocol with epoch horizon 300 |
| Ours CIFAR + Ours Flowers + ALG Flowers | `/app/output/researcher_sync_ours_alg_300ep_seed1` | 1 / researcher sync |

No current 300-epoch result should replace a historical file in place. After
collection and verification, it must be added with its epoch, seed, and
protocol family in the protocol-ID directory. See `PENDING_IMPORTS.md` for the
exact expected destinations of the jobs currently running.

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

Generic methods use the CNN-to-ViT adapters documented in each method
directory. They should not be described as unmodified original CNN-to-CNN
experiments. Ours is maintained separately from the five generic baselines.
