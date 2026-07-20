# Teacher stage

This directory contains the complete 32 x 32 ResNet56 teacher stage used
before downstream CNN-to-ViT experiments.

```text
teachers/
├── checkpoints/
│   ├── cifar100/
│   ├── flowers102/
│   ├── chaoyang/
│   ├── README.md
│   └── manifest.json
├── train_teacher_cifar100.py
├── train_teacher_flowers.py
├── train_teacher_chaoyang.py
└── verify_checkpoints.py
```

The training recipes are documented in the repository-level `PROTOCOL.md`.
The fixed weights, metrics, configs, summaries, logs, and integrity hashes are
documented in `checkpoints/README.md`.

Verify all selected teacher checkpoints from the repository root:

```bash
python teachers/verify_checkpoints.py --dataset all
```
