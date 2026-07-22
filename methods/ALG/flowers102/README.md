# ALG: Flowers-102 / DeiT-Ti

This entry point applies the researcher-supplied ALG configuration and
controller to Flowers-102 without changing the method-specific logic.

| Setting | Value |
|---|---:|
| Teacher | fixed ResNet56, native `32 x 32` |
| Student | DeiT-Ti from scratch, `224 x 224` |
| Current split | official train `1,020` / val `1,020` / test `6,149` |
| Checkpoint selection | highest official-val Top-1 |
| Reported result | selected best checkpoint on official test, once |
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
python methods/ALG/flowers102/train_official_split.py --timing-run --num-workers 4
```

Full run:

```bash
python methods/ALG/flowers102/train_official_split.py --num-workers 4 \
  --run-name alg_flowers102_deit_ti_official_split_300ep_seed1 \
  --output-dir /app/output
```

The older `train.py` entry point intentionally preserves the earlier
`train+val -> test-best` researcher-sync-v1 behavior for provenance only.
Do not use it for the new official-three-way result. Historical runs remain
separately labeled under `results/` and are never overwritten.
