# Distillation methods

This directory contains the V2 ResNet56-to-DeiT-Ti student pipelines for the
five generic KD baselines in the draft table: KD, CRD, ReviewKD, MGD, and OFA.
Every method uses the same fixed teacher checkpoint and the same base student
protocol for a given dataset. Only the documented transfer operator and its
method-specific coefficients may differ.

## Locked common student protocols

| Dataset | Epochs | Batch | Warm-up | Student input | Teacher input |
|---|---:|---:|---:|---:|---:|
| CIFAR-100 | 300 | 128 | 20 | 224 x 224 | 32 x 32 |
| Flowers-102 | 200 | 64 | 5 | 224 x 224 | 32 x 32 |
| Chaoyang | 100 | 64 | 5 | 224 x 224 | 32 x 32 |

All datasets use scratch DeiT-Ti (`deit_tiny_patch16_224`), AdamW with initial
learning rate `5e-4` and weight decay `0.05`, cosine decay after warm-up, label
smoothing `0.1`, CUDA AMP, seed `42`, and test Top-1 evaluation. No external
student pretrained weights are used.

Training uses random resized crop to 224 (`scale=0.8-1.0`, bicubic) and random
horizontal flip. Evaluation uses resize to 256 and center crop to 224.
CIFAR-100 uses CIFAR normalization for the student; Flowers and Chaoyang use
ImageNet normalization.

The teacher does **not** receive an independently augmented image. The exact
student view is converted back to image space, bilinearly resized from 224 to
32, and normalized with the ImageNet statistics used to train the fixed
teacher. Crop and flip geometry therefore remains shared across both branches.

## Method-specific operators

| Method | Transfer | V2 CNN-to-ViT connection |
|---|---|---|
| KD | class logits | no spatial adapter |
| CRD | pooled representation | ResNet stage-3 GAP `64d`; DeiT CLS pre-logits `192d` |
| ReviewKD | multi-level features | ResNet 32/16/8 grids bilinearly resized to DeiT 14x14 grid |
| MGD | masked reconstruction | ResNet stage-3 8x8 bilinearly resized to 14x14; DeiT block-11 tokens; `192 -> 64` alignment |
| OFA | projected class logits | DeiT blocks 1/3/9/11 and official-behavior transformer projectors |

Method settings and official-code provenance are recorded under each method
directory. The CNN-to-ViT adapters are explicit V2 implementation choices;
they are not presented as the original CNN-to-CNN configurations.

## Five-method timing and full execution

The generic runner executes each selected method as a separate Python
subprocess in canonical order:

```bash
python methods/run_five_methods.py --dataset flowers102 --timing-run \
  --output-dir /app/output/flowers102_five_methods_timing_v2 --num-workers 4

python methods/run_five_methods.py --dataset chaoyang --timing-run \
  --output-dir /app/output/chaoyang_five_methods_timing_v2 --num-workers 4
```

To collect all ten timings in one Issue and one combined summary:

```bash
python methods/run_flowers_chaoyang_timing.py \
  --num-workers 4
```

This upper-level runner writes `two_dataset_timing_status.json` and
`two_dataset_timing_summary.json`, while preserving both per-dataset summaries
for the lifetime of the timing Pod. Every required estimate is also printed.

Every method has its own run directory and writes `student_best.pt`,
`student_latest.pt`, and `summary.json`. The runner writes
`five_method_status.json` while running and `five_method_summary.json` after
success. A later failure stops the sequence but does not delete artifacts from
already completed methods.

The final timing log reports each method's average epoch time and full-run
estimate plus the combined estimate. Use an ordered subset when the measured
total would leave insufficient margin under the 600-minute Pod limit:

```bash
python methods/run_five_methods.py --dataset flowers102 --full-run \
  --methods KD CRD ReviewKD --output-dir /app/output/flowers102_group1_v2 \
  --num-workers 4

python methods/run_five_methods.py --dataset flowers102 --full-run \
  --methods MGD OFA --output-dir /app/output/flowers102_group2_v2 \
  --num-workers 4
```

The exact grouping must be chosen from the returned H200 timing log. A safe
target is at most about 540 minutes per Issue, leaving roughly one hour for
downloads, setup, evaluation, checkpoint writes, and runtime variance.

## Measured combined full batch

H200 build 449 measured Flowers five methods at `3h 20m 01s` and Chaoyang five
methods at `1h 19m 30s`. Appending the measured CIFAR-100 KD estimate of
`3h 06m 28s` gives `7h 45m 59s`, which retains `2h 14m 01s` below the
600-minute limit.

```bash
python methods/run_combined_full_batch.py --cifar-method KD \
  --output-dir /app/output/combined_flowers_chaoyang_cifar100_kd_v2 \
  --num-workers 4
```

Execution is short-first: Chaoyang all five, Flowers all five, then CIFAR-100
KD. This maximizes the number of completed results preserved if a later task
fails. The runner also accepts the other already measured CIFAR choices `CRD`
and `ReviewKD`; all three plans retain more than two hours below the Pod limit.
