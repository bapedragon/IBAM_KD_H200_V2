# Ours V1: teacher CNN to transformer spatial collapse

This is an isolated experiment requested after reviewing the intended
spatial-structure ablation. It is not the old grid-permutation or token-space
implementation, and it does not modify `methods/Ours/ours.py`.

## Direction of conversion

The original Ours V1 aligns the student to each teacher CNN stage:

```text
student [B,192,14,14]
  -> Conv1x1 192->16/32/64
  -> resize to 32x32 / 16x16 / 14x14
```

This experiment reverses that direction:

```text
teacher [B,16/32/64,H,W]
  -> bilinear resize to the student 14x14 patch grid
  -> Conv1x1 16/32/64->192
  -> flatten [B,192,14,14] to [B,196,192]
```

The aggregated student features are also flattened to `[B,196,192]`.
Content-only V1 cross-attention, `L_align`, and `L_fuse` are then evaluated in
that common transformer representation. There is no permutation, no
stage-specific target grid, and no explicit 2D positional term.

The flatten operation removes explicit 2D geometry from the fusion interface;
it does not discard tensor values and should not be described as a lossy
operation by itself.

## Locked CIFAR-100 command

```bash
python methods/Ours/cnn_to_transformer_spatial_collapse/train_cifar100.py \
  --num-workers 4 \
  --output-dir /app/output \
  --run-name ours_v1_cnn_to_transformer_spatial_collapse_300ep_seed1
```

Run with `--timing-run` before a full 300-epoch job.
