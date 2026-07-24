# Ours V1 CNN-to-transformer spatial-collapse review snapshot

[`train_cifar100_full.py`](train_cifar100_full.py) expands the complete
model-side experiment in one reviewable file. It does not import model or loss
code from the modular implementation.

The key reversal is visible directly in `__init__`:

```text
Original Ours V1:
  Conv1x1 192 -> 16/32/64

This experiment:
  Conv1x1 16/32/64 -> 192
```

For all three teacher stages, the forward path is:

```text
teacher BCHW
  -> bilinear resize to the student's 14x14 patch grid
  -> Conv1x1 to D=192
  -> flatten to BND=[B,196,192]
  -> content-only Ours V1 cross-attention and token-representation losses
```

There is no grid permutation, no stage-specific target grid, and no explicit
2D positional bias. The original `methods/Ours/ours.py` remains untouched.

Generic dataset loading, the frozen teacher and DeiT construction, optimizer,
scheduler, evaluation, and checkpoint I/O still use `methods/Ours/core.py` to
avoid a second divergent trainer.

## Run

```bash
python serve/ours_v1_cnn_to_transformer_spatial_collapse/train_cifar100_full.py \
  --timing-run \
  --num-workers 4 \
  --output-dir /app/output \
  --run-name ours_v1_cnn_to_transformer_spatial_collapse_timing_2ep
```
