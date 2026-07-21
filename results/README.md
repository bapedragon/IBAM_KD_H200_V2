# Curated student results

This directory is the compact, reporting-oriented view of completed H200
student runs. Raw Pod output folders contained repeated batch wrappers and both
`best` and `latest` checkpoints. Here they are normalized to:

```text
results/
├── KD/{cifar100,flowers102,chaoyang}/
├── CRD/{cifar100,flowers102,chaoyang}/
├── ReviewKD/{flowers102,chaoyang}/
├── MGD/{flowers102,chaoyang}/
├── OFA/{flowers102,chaoyang}/
├── Ours/cifar100/
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

All 13 committed checkpoints were loaded with PyTorch and verified against
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
| ReviewKD | - | - | - | - | Pending |
| MGD | - | - | - | - | Pending |
| OFA | - | - | - | - | Pending |

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

Fixed protocol: ResNet56 teacher at 32 x 32, scratch DeiT-Ti student at
224 x 224, 100 epochs, seed 42.

| Method | Best epoch | Best Top-1 | Last Top-1 | Vanilla gap |
|---|---:|---:|---:|---:|
| Vanilla DeiT-Ti | - | 82.00% | - | - |
| KD | 15 | **62.79%** | 56.80% | -19.21 pp |
| CRD | 61 | **79.66%** | 77.93% | -2.34 pp |
| ReviewKD | 86 | **81.72%** | 81.07% | -0.28 pp |
| MGD | 80 | **80.69%** | 79.94% | -1.31 pp |
| OFA | 90 | **75.55%** | 74.99% | -6.45 pp |

## Source runs

- `run_logs/h200_build-450_combined-generic-kd.log`: Chaoyang five methods,
  Flowers-102 five methods, then CIFAR-100 KD.
- `run_logs/h200_build-452_cifar100-ours-crd.log`: CIFAR-100 Ours and CRD.

Generic methods use the CNN-to-ViT adapters documented in each method
directory. They should not be described as unmodified original CNN-to-CNN
experiments. Ours is maintained separately from the five generic baselines.
