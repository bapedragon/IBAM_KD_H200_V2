# ALG: CIFAR-100 / DeiT-Ti (researcher sync)

This entry point applies the researcher-supplied ALG configuration and
controller to CIFAR-100. It is the direct ALG comparison for the synchronized
Ours run and does not reuse the older draft-common recipe.

| Setting | Value |
|---|---:|
| Teacher | fixed ResNet56, native `32 x 32` |
| Student | DeiT-Ti from scratch, `224 x 224` |
| Split | official train / official test |
| Epochs | 300 |
| Train / eval batch | 64 / 200 |
| Optimizer | AdamW |
| LR / minimum LR | `5e-4` / `5e-6` |
| Weight decay | `0.05` |
| Warm-up | 20 epochs, factor `0.001` |
| Schedule | cosine |
| Label smoothing / drop path | `0.0` / `0.1` |
| Seed / precision | 1 / FP32 |
| Augmentation | public LG strong augmentation |
| ALG beta / threshold / window | `2.5` / `-0.02` / 50 |
| Grid alignment | larger of teacher/student, bilinear (`32/16/14`) |

Timing run:

```bash
python methods/ALG/cifar100/train.py --timing-run --num-workers 4
```

Full run:

```bash
python methods/ALG/cifar100/train.py --num-workers 4 \
  --run-name alg_cifar100_deit_ti_researcher_sync_300ep_seed1 \
  --output-dir /app/output
```

Every setting above is locked by the dataset wrapper. Command-line overrides
remain possible for controlled ablations, but must be recorded as a different
protocol and run name.
