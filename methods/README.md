# Distillation methods

This directory starts the V2 student stage. Every method must use the same
fixed CIFAR-100 teacher checkpoint and the same base DeiT-Ti protocol. A method
may change only its explicitly documented distillation operator and associated
coefficient.

## Fixed CIFAR-100 student protocol

| Item | Value |
|---|---:|
| Student | DeiT-Ti (`deit_tiny_patch16_224`) |
| Initialization | scratch; no external pretrained weights |
| Student input | **224 x 224** |
| Teacher | fixed ResNet56 checkpoint from `teachers/checkpoints/manifest.json` |
| Teacher input | **32 x 32** |
| Epochs | 300 |
| Batch size | 128 |
| Optimizer | AdamW |
| Initial LR | `5e-4` |
| Weight decay | `0.05` |
| LR schedule | 20-epoch warm-up, then cosine decay |
| Label smoothing | `0.1` |
| AMP | enabled on CUDA |
| Seed | 42 |
| Evaluation | CIFAR-100 test Top-1 |

The student transform is random resized crop to 224 (`scale=0.8-1.0`, bicubic),
random horizontal flip, tensor conversion, and CIFAR-100 normalization. Test
images use resize to 256, center crop to 224, and the same normalization.

The teacher does **not** receive an independently augmented image. The exact
student tensor is converted back to image space, bilinearly resized from 224
to 32, and normalized with the ImageNet statistics used to train the fixed
teacher. Crop and flip geometry therefore remains shared across both branches.

## Method-specific operators

| Method | Transfer | V2 adaptation |
|---|---|---|
| KD | class logits | No spatial adapter required |
| CRD | pooled representation | ResNet stage-3 GAP `64d`; DeiT CLS pre-logits `192d` |
| ReviewKD | multi-level features | ResNet 32/16/8 grids bilinearly resized to DeiT 14x14 grid |

Method-specific settings and official-code provenance are recorded under each
method directory. The generic CNN-to-ViT adapters are explicit V2
implementation choices; they are not claimed to be the original CNN-to-CNN
configurations from the corresponding papers.

## Sequential execution

The runner executes `KD -> CRD -> ReviewKD` as separate Python subprocesses:

```bash
python methods/run_cifar100_three_methods.py --timing-run --num-workers 4
```

Every subprocess has an independent run name and directory. A failure stops
the sequence and records `three_method_status.json`; outputs from already
completed methods remain intact. After all three finish,
`three_method_summary.json` lists their individual result directories.

The ordered subset can be selected when the combined estimate is too close to
the Pod runtime limit:

```bash
python methods/run_cifar100_three_methods.py --full-run --methods KD CRD \
  --output-dir /app/output/cifar100_kd_crd_full_v2 --num-workers 4

python methods/run_cifar100_three_methods.py --full-run --methods ReviewKD \
  --output-dir /app/output/cifar100_reviewkd_full_v2 --num-workers 4
```

The legacy aggregate filenames remain `three_method_status.json` and
`three_method_summary.json` for compatibility even when a subset is selected;
the JSON payload records the actual `selected_methods`.
