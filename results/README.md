# Curated student results

This directory is the compact, reporting-oriented view of completed H200
student runs. Raw Pod output folders contained repeated batch wrappers and both
`best` and `latest` checkpoints. Here they are normalized to:

```text
results/
├── KD/{cifar100,flowers102,chaoyang}/
├── CRD/{cifar100,flowers102,chaoyang}/
├── ReviewKD/{cifar100,flowers102,chaoyang}/
├── MGD/{cifar100,flowers102,chaoyang}/
├── OFA/{cifar100,flowers102,chaoyang}/
├── ALG/chaoyang/
├── Ours/{cifar100,chaoyang}/
├── run_logs/
└── CHECKSUMS.sha256
```

Each completed method/dataset folder contains `run_summary.json` and the
selected `student_deit_ti_best_top1-XX.XX.pt` checkpoint. Account names and
H200 build numbers are kept only under `run_logs`; they do not determine
checkpoint placement.

Only the selected best checkpoint is committed. The original downloaded
outputs retain `student_latest.pt`; its exact final-epoch accuracy is also
preserved in `run_summary.json`. This avoids doubling repository size and H200
clone time without discarding the reported result.

All 18 committed checkpoints were loaded with PyTorch and verified against
their summaries for dataset, method, best accuracy, and checkpoint epoch.
Checkpoint file names show Top-1 rounded to two decimals; summaries preserve
full precision.

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

## Flowers-102

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

## Chaoyang

All rows use the fixed ResNet56 teacher at 32 x 32 and a scratch DeiT-Ti
student at 224 x 224. The five generic methods and the historical Ours run use
100 epochs and seed 42. The standalone ALG reproduction uses the audited
public LG/ALG base: 300 epochs, batch 128, 20-epoch warm-up, FP32, and seed 1.

| Method | Best epoch | Best Top-1 | Last Top-1 | Vanilla gap | Status |
|---|---:|---:|---:|---:|---|
| Vanilla DeiT-Ti | - | 82.00% | - | - | Draft reference |
| ALG | 235 | **80.32%** | 79.71% | -1.68 pp | Verified public LG/ALG-base run |
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
