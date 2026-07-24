# Serve

This directory contains readable snapshots of experiments that otherwise span
multiple reusable modules in `methods/`.

These files do not replace the maintained method implementations. They collect
the important model and intervention logic in one place so an experiment can
be reviewed without following several imports.

- [`table4_grid_permutation/`](table4_grid_permutation/): the exact Table 4
  fixed teacher-grid permutation experiment that produced 81.79% Top-1.
- [`ours_v2/`](ours_v2/): base Ours V2 with aggregation, deformable
  enhancement, position-aware cross-attention, both feature losses, and the
  locked CIFAR-100 protocol expanded in one reviewable file.
- [`ours_v1_cnn_to_transformer_spatial_collapse/`](ours_v1_cnn_to_transformer_spatial_collapse/):
  isolated V1 experiment that reverses feature alignment by projecting each
  teacher CNN stage into the student's flattened transformer representation.
