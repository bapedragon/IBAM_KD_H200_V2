# H200 outputs awaiting artifact import

This file is an inbox, not a result table. A row means that an H200 job was
submitted or completed but its output archive has not yet been received and
verified in this repository. Accuracy cells intentionally remain blank.

## Current status

There are no pending imports in the overnight batch received on 2026-07-22.
The following families were loaded with PyTorch, checked against their JSON
summaries, and imported without replacing historical results:

- `generic_kd_300ep_epoch_only_v1_seed42`: five Flowers-102 and five
  Chaoyang generic-KD runs;
- `researcher_sync_v1_300ep_seed1`: Ours CIFAR-100, Ours Flowers-102, and
  ALG Flowers-102;
- `researcher_sync_v2_official_three_way_300ep_seed1_historical`: the
  Flowers train/validation/test audit retained for provenance.

Add new jobs below this section only after submission, and remove their rows
after the same import gate has passed.

## Import gate

Before moving a pending output into `results/`:

1. load `student_best.pt` with PyTorch;
2. compare method, dataset, epoch, Top-1, seed, and protocol arguments with
   `summary.json`;
3. verify the teacher SHA-256 against `teachers/checkpoints/manifest.json`;
4. place it only in its exact protocol-ID directory;
5. update the result table and `CHECKSUMS.sha256`;
6. change or remove the corresponding pending row only after verification.
