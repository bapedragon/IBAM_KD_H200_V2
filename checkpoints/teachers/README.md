# Fixed 32 x 32 teacher checkpoints

These are the primary ResNet56 teacher weights selected after the full H200
runs. The same checkpoint must be reused for every compared KD method on the
corresponding dataset.

| Dataset | Selected checkpoint | H200 build | Epoch | Top-1 | Draft reference | Gap |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-100 | `cifar100/teacher_resnet56_cifar100_32_best.pt` | 438 | 300 | 71.91% | 70.43% | +1.48 pp |
| Flowers-102 | `flowers102/teacher_resnet56_flowers102_32_best.pt` | 447 | 389 | 66.03% | 66.33% | -0.30 pp |
| Chaoyang | `chaoyang/teacher_resnet56_chaoyang_32_best.pt` | 443 | 94 | 76.72% | 77.20% | -0.48 pp |

Every dataset directory also contains:

- `training_config.json`: locked settings written before training;
- `metrics.csv`: epoch-by-epoch measurements;
- `training_summary.json`: final result, paths, timing, protocol, and hashes;
- `training_log.txt`: full H200 issue output.

Only the primary `best` checkpoint is committed. The locally archived H200
output still retains `best`, `latest`, and `closest_to_reference` files.

Verify hashes, metadata, strict state-dict loading, and a 32 x 32 forward pass:

```bash
python teacher_checkpoints.py --dataset all
```

The datasets are not included. Chaoyang remains mounted separately at
`/app/data/chaoyang`.
