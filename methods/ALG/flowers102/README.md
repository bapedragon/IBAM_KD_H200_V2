# ALG: Flowers-102 / DeiT-Ti

The current official-split entry point isolates ALG from Ours. It uses the
ALG-paper optimization/method protocol, the public LG feature path, and the
adaptive schedule defined by the ALG equations. No Ours configuration or Ours
module is used.

One split distinction is explicit: the ALG paper describes Flowers as `2,040`
training images and `6,149` test images, whereas this matched comparison locks
the official `1,020` train / `1,020` val / `6,149` test split requested for
both methods. This is a shared evaluation choice, not an Ours setting injected
into ALG.

| Setting | Value |
|---|---:|
| Teacher | fixed ResNet56, native `32 x 32` |
| Student | DeiT-Ti from scratch, `224 x 224` |
| Current split | official train `1,020` / val `1,020` / test `6,149` |
| Checkpoint selection | highest official-val Top-1 |
| Reported result | selected best checkpoint on official test, once |
| Epochs | 300 |
| Train / eval batch | 128 / 200 |
| Optimizer | AdamW |
| LR / minimum LR | `5e-4` / `5e-6` |
| Weight decay | `0.05` |
| Warm-up | 20 epochs, factor `0.001` |
| Schedule | cosine |
| Label smoothing / drop path | `0.0` / `0.1` |
| Seed / precision | 1 / FP32 |
| Augmentation | public LG strong augmentation |
| ALG beta / threshold / window | `2.5` / `-0.02` / 50 |
| Controller stop warm-up | none; evaluate as soon as the derivative exists |
| Stop condition | paper Eq. (19): stop at `smoothed_derivative >= tau` |
| Early derivative | ALG Eq. (16), explicit `1/e` normalization |
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
researcher-sync experiment for provenance only. It used train batch 64 and a
20-epoch controller stop warm-up and must not be reported as the paper-faithful
ALG result. Historical runs remain separately labeled and are never
overwritten.
