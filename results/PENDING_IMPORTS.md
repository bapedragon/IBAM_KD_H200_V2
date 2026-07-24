# H200 outputs awaiting artifact import

This file is an inbox, not a result table. A row means that an H200 job was
submitted or completed but its output archive has not yet been received and
verified in this repository. Accuracy cells intentionally remain blank.

## Current status

The overnight batch received on 2026-07-22 and builds 479, 480, 482, 484,
485, 488, 490, 493, 494, 496, and 503 received through 2026-07-24 have been
imported. One completed run still awaits its checkpoint archive:

| Method | Dataset | Protocol ID | Log-verified result | Expected destination | Missing artifacts |
|---|---|---|---:|---|---|
| Ours | Chaoyang | `researcher_sync_v1_300ep_seed1` | 81.95% | `results/Ours/chaoyang/researcher_sync_v1_300ep_seed1/` | `student_best.pt`, `run_summary.json` |

The completed H200 log reports best epoch 292, last Top-1 81.11%, guidance
stop epoch 193, and selected best Top-1 81.95%. This value may be shown with a
pending-artifact marker, but it is not counted among the committed and
PyTorch-verified checkpoints.

The following received families were loaded with PyTorch, checked against
their JSON summaries, and imported without replacing historical results:

- `generic_kd_300ep_epoch_only_v1_seed42`: five Flowers-102 and five
  Chaoyang generic-KD runs;
- `researcher_sync_v1_300ep_seed1`: Ours CIFAR-100, Ours Flowers-102, and
  ALG Flowers-102;
- `researcher_sync_v2_official_three_way_300ep_seed1_historical`: the
  Flowers train/validation/test audit retained for provenance.
- `table4_grid_permuted_researcher_sync_v1_300ep_seed1_permseed1`: Table 4
  grid-permutation best checkpoint and summary;
- `table7_lambda_0_researcher_sync_v1_300ep_seed1`: Table 7 lambda-zero best
  checkpoint and summary;
- `table7_lambda_0p25_researcher_sync_v1_300ep_seed1`: Table 7 lambda-0.25
  best checkpoint and summary.
- `table7_lambda_0p75_researcher_sync_v1_300ep_seed1` and
  `table7_lambda_1_researcher_sync_v1_300ep_seed1`: the remaining received
  Table 7 convex-sweep checkpoints and summaries;
- `table4_kv_independent_researcher_sync_v1_300ep_seed1_k1_v1001`,
  `table4_local_patch2_researcher_sync_v1_300ep_seed1_permseed1`, and
  `table4_token_space_researcher_sync_v1_300ep_seed1`: the three received
  Table 4 follow-up controls;
- `table7_lambda_0_relative_position_v1_300ep_seed1` and
  `table7_lambda_0p5_relative_position_v1_300ep_seed1`: the paired Ours V2
  relative-position Table 7 runs from build 503;
- `paper_lg_v2_trainval_test_b128_300ep_seed1`: selected ALG Flowers batch-128
  result from build 479;
- `paper_source_v2_trainval_test_b128_300ep_seed1`: auxiliary Ours Flowers
  batch-128 result from build 479;
- `paper_lg_v2_trainval_test_b64_300ep_seed1`,
  `paper_lg_v2_b128_300ep_seed1`, and `paper_lg_v2_b64_300ep_seed1`: pure-ALG
  batch controls from build 480;
- `cifar100_locked_b64_v1_300ep_seed1`: auxiliary Ours Chaoyang batch-64
  result from build 480.

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
