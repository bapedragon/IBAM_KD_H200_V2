# Researcher-code synchronization audit (2026-07-21)

This file records exactly what changed after comparing the repository with the
researcher screenshots and the originally supplied Ours module.

## Already matching; retained

- all 12 DeiT features are aggregated into three learned mixtures;
- student stages are projected by stage-specific `1 x 1` convolutions;
- teacher and student features are bilinearly resized to the larger grid;
- the CCC attention block receives projected student and teacher features;
- guidance loss is the sum over stages of
  `0.5*MSE(teacher, aligned) + 0.5*MSE(teacher, fused)`;
- the trainer applies `CE + beta*guidance_loss`, so `beta=2.5` is applied once;
- logit distillation is disabled for the shown configuration.

## Changed from historical repository behavior

| Item | Historical runs | Researcher-synchronized runs |
|---|---|---|
| Controller observation | epoch mean of `L_align` | epoch mean of complete `0.5*L_align+0.5*L_fuse` |
| Controller warm-up | no explicit stop warm-up | no stop before epoch 20 |
| Stop safety | required an observed descent first | no descent-first guard |
| Threshold comparison | `>= tau` after descent | strict `> tau` |
| Early derivative | included an epoch-1 zero | researcher loops start at derivative index 2 |
| Chaoyang grid | historical diagnostic `32/16/8` | executable larger-grid `32/16/14` |
| Chaoyang train batch | 128 | 64, following the shown researcher config |

The old behavior is preserved under `legacy/` and in Git commit `ee2dc55`.

## Remaining screenshot boundary

The screenshots do not show the line that calls `update_epoch()`. The active
standalone adapter uses human-readable one-based epochs (`1..300`), which is
consistent with the controller formulas and printed epoch numbers. If the
original pycls trainer passed zero-based `cur_epoch` without adding one, that
would create a one-epoch/index lag and should be confirmed from the missing
call-site line before claiming byte-identical execution. This does not block a
timing run, and the chosen numbering is printed/saved in the run metadata.
