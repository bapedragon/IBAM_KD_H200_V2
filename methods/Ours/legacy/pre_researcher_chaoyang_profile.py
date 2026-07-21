"""Archived Chaoyang defaults used before researcher-code synchronization."""

PROTOCOL_DEFAULTS = (
    ("--protocol-name", "chaoyang_deit_ti_ours_draftgrid_algbase_v3"),
    ("--student-epochs", "300"),
    ("--batch-size", "128"),
    ("--lr", "0.0005"),
    ("--min-lr", "0.000005"),
    ("--weight-decay", "0.05"),
    ("--warmup-epochs", "20"),
    ("--warmup-factor", "0.001"),
    ("--label-smoothing", "0.0"),
    ("--drop-path-rate", "0.1"),
    ("--eval-batch-size", "200"),
    ("--seed", "1"),
    ("--base-protocol", "lg_official"),
    ("--teacher-image-size", "32"),
    ("--beta-schedule", "alg"),
    ("--beta-on", "2.5"),
    ("--alg-threshold", "-0.02"),
    ("--alg-smoothing-window", "50"),
    ("--grid-resize-mode", "teacher"),
    ("--eval-resize-mode", "direct"),
)
