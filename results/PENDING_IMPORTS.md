# H200 outputs awaiting artifact import

This file is an inbox, not a result table. A row means that an H200 job was
submitted or completed but its output archive has not yet been received and
verified in this repository. Accuracy cells intentionally remain blank.

## Current status

The overnight batch received on 2026-07-22 has been imported. Two completed
runs still await their checkpoint archives:

| Method | Dataset | Protocol ID | Log-verified result | Expected destination | Missing artifacts |
|---|---|---|---:|---|---|
| Ours | Chaoyang | `researcher_sync_v1_300ep_seed1` | 81.95% | `results/Ours/chaoyang/researcher_sync_v1_300ep_seed1/` | `student_best.pt`, `run_summary.json` |
| ALG | Flowers-102 | `flowers102_deit_ti_alg_paper_lg_v2_trainval_test` | 73.15% | `results/ALG/flowers102/alg_paper_lg_v2_trainval_test_300ep_seed1/` | `student_best.pt`, `summary.json` or normalized `run_summary.json` |

The completed H200 log reports best epoch 292, last Top-1 81.11%, guidance
stop epoch 193, and selected best Top-1 81.95%. This value may be shown with a
pending-artifact marker, but it is not counted among the 33 committed and
PyTorch-verified checkpoints.

The supplied final excerpt for the pure ALG Flowers run reports selected best
Top-1 `73.15%` under train/eval batch `128/200`, 300 epochs, train+val/test,
and seed 1. The excerpt does not contain ALG's best epoch or last-epoch
accuracy, so those fields remain unknown until the output archive is received.

The following received families were loaded with PyTorch, checked against
their JSON summaries, and imported without replacing historical results:

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
