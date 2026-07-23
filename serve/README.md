# Serve

This directory contains readable snapshots of experiments that otherwise span
multiple reusable modules in `methods/`.

These files do not replace the maintained method implementations. They collect
the important model and intervention logic in one place so an experiment can
be reviewed without following several imports.

- [`table4_grid_permutation/`](table4_grid_permutation/): the exact Table 4
  fixed teacher-grid permutation experiment that produced 81.79% Top-1.
