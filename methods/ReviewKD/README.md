# ReviewKD

The ABF and hierarchical context loss follow the authors' official ReviewKD
implementation behavior.

- Official repository: https://github.com/dvlab-research/ReviewKD
- Pinned commit: `cede6ea6387ae9b6127de0e561507177bf19c11e`
- Relevant official files: `CIFAR-100/model/reviewkd.py`,
  `CIFAR-100/train.py`, and `CIFAR-100/script/reviewKD.sh`

## Objective

```text
L = CE + ramp(epoch, 20) * 0.6 * HCL
```

The official CIFAR-100 command supplies feature-loss weight `0.6`; the feature
term ramps linearly over 20 epochs. No additional logit KL term is used.

## CNN-to-ViT adapter

- Teacher: post-activation ResNet56 stages 1/2/3, grids 32/16/8 and channels
  16/32/64.
- Student: DeiT-Ti transformer blocks 3/7/11, patch grid 14x14 and 192
  channels.
- Spatial bridge: bilinear interpolation of each teacher feature to 14x14,
  `align_corners=False`.
- Channel/fusion bridge: official-behavior ABF, hidden width 192, deep-to-
  shallow fusion.
- HCL: full 14x14 grid plus pooled 4x4, 2x2, and 1x1 comparisons.

The ABF adapter state is stored with the student in every checkpoint. The
cross-architecture feature selection and bilinear bridge are documented V2
choices; base optimization and input settings are in `methods/README.md`.
