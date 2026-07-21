# ALG: Flowers-102 / DeiT-Ti (researcher sync)

This entry point applies the researcher-supplied ALG configuration and
controller to Flowers-102 without changing the method-specific logic.

| Setting | Value |
|---|---:|
| Teacher | fixed ResNet56, native `32 x 32` |
| Student | DeiT-Ti from scratch, `224 x 224` |
| Split | official train+val / official test |
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
| Grid alignment | larger of teacher/student, bilinear |

Timing run:

```bash
python methods/ALG/flowers102/train.py --timing-run --num-workers 4
```

Full run:

```bash
python methods/ALG/flowers102/train.py --num-workers 4 --run-name alg_flowers102_deit_ti_researcher_sync_300ep_seed1 --output-dir /app/output
```

The older generic Flowers runs remain separately labeled as 200-epoch,
seed-42 historical results under `results/`; they are not overwritten.
