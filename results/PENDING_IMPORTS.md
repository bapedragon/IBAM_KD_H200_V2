# H200 outputs awaiting artifact import

This file is an inbox, not a result table. A row means that an H200 job was
submitted or completed but its output archive has not yet been received and
verified in this repository. Accuracy cells intentionally remain blank.

## Current researcher-sync batch

Collection root on H200:
`/app/output/researcher_sync_ours_alg_300ep_seed1`

| Method | Dataset | Protocol ID | H200 state | Repository destination after verification |
|---|---|---|---|---|
| Ours | CIFAR-100 | `researcher_sync_v1_300ep_seed1` | running | `results/Ours/cifar100/researcher_sync_v1_300ep_seed1/` |
| Ours | Flowers-102 | `researcher_sync_v1_300ep_seed1` | running | `results/Ours/flowers102/researcher_sync_v1_300ep_seed1/` |
| ALG | Flowers-102 | `researcher_sync_v1_300ep_seed1` | running | `results/ALG/flowers102/researcher_sync_v1_300ep_seed1/` |

The timing run completed all three tasks and estimated `7h 56m 57s`, leaving
`2h 03m 03s` below the 600-minute Pod limit. Timing-run accuracies are not
research results and must not be imported here.

## Generic-KD epoch-only reruns

The previously submitted Flowers/Chaoyang 300-epoch reruns use the separate
`generic_kd_300ep_epoch_only_v1_seed42` family. They retain the generic-KD
batch, warm-up, augmentation, AMP, seed, adapters, and losses; only the epoch
and cosine horizon change to 300.

Expected destinations are:

```text
results/<KD|CRD|ReviewKD|MGD|OFA>/<flowers102|chaoyang>/
  generic_kd_300ep_epoch_only_v1_seed42/
```

These files must not replace the historical Flowers 200-epoch or Chaoyang
100-epoch directories.

## Import gate

Before moving a pending output into `results/`:

1. load `student_best.pt` with PyTorch;
2. compare method, dataset, epoch, Top-1, seed, and protocol arguments with
   `summary.json`;
3. verify the teacher SHA-256 against `teachers/checkpoints/manifest.json`;
4. place it only in its exact protocol-ID directory;
5. update the result table and `CHECKSUMS.sha256`;
6. change or remove the corresponding pending row only after verification.
