# Masked Generative Distillation (MGD)

This folder applies MGD to the fixed ResNet56-to-DeiT-Ti comparison. The
masking, alignment, generator, reduction, and method coefficients follow the
authors' official image-classification implementation.

## Official-code provenance

- Paper: Masked Generative Distillation, ECCV 2022
- Official repository: https://github.com/yzd-v/MGD
- Pinned commit: `2c9da0b28625eb948db57afc02c824452c3910fe`
- Official source: `cls/mmcls/distillation/losses/mgd.py`
- Official classification config:
  `cls/configs/distillers/mgd/res34_distill_res18_img.py`
- License: Apache-2.0, copied to `OFFICIAL_CODE_LICENSE.txt`

`official_mgd.py` removes only the old MMClassification registry dependency
and adds explicit shape checks. It preserves the official classification loss:

```text
total loss = classification CE + alpha * summed reconstruction MSE / batch
alpha = 0.00007
mask probability = 0.15
```

The official classification code samples a mask of shape `N x C x 1 x 1`.
Consequently, this implementation masks channels consistently across spatial
locations; it does not silently replace the official behavior with a spatial
patch mask. No logit KL loss is added.

## CNN-to-ViT adapter

MGD consumes one spatial feature map from each network. The heterogeneous
connection is fixed as follows:

- Teacher: post-activation ResNet56 `stage3`, channel dimension `64`
- Teacher spatial adapter: bilinear interpolation from `8 x 8` to `14 x 14`,
  `align_corners=False`
- Student: DeiT-Ti block 11 patch tokens, reshaped by timm to `14 x 14`
- Student channel dimension: `192`
- Official MGD alignment: trainable `1 x 1` convolution, `192 -> 64`
- Official generator: `3 x 3 Conv -> ReLU -> 3 x 3 Conv`

This adapter is the only CNN-to-ViT modification. Feature shapes are checked
before training, and the teacher remains frozen.

## Fixed dataset protocols

| Dataset | Epochs | Batch | Optimizer | LR | Weight decay | LR warm-up |
|---|---:|---:|---|---:|---:|---:|
| CIFAR-100 | 300 | 128 | AdamW | `5e-4` | `0.05` | 20 |
| Flowers-102 | 200 | 64 | AdamW | `5e-4` | `0.05` | 5 |
| Chaoyang | 100 | 64 | AdamW | `5e-4` | `0.05` | 5 |

All use cosine decay, 224-pixel inputs, label smoothing `0.1`, AMP, seed `42`,
no external pretraining, and the same dataset split and fixed teacher used by
KD, CRD, and ReviewKD.

## Timing runs

```bash
python methods/MGD/cifar100/train.py --timing-run --num-workers 4

python methods/MGD/flowers102/train.py --timing-run --num-workers 4

python methods/MGD/chaoyang/train.py --timing-run \
  --data-dir /app/data/chaoyang --num-workers 4
```

For the combined five-method timing and safe full-run grouping, use
`methods/run_five_methods.py` as documented in `methods/README.md`.

Example individual CIFAR-100 full run after timing verification:

```bash
python methods/MGD/cifar100/train.py --student-epochs 300 --num-workers 4 --run-name mgd_cifar100_deit_ti_300ep --output-dir /app/output
```
