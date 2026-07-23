# Ours: CUB-200-2011 / DeiT-Ti

This folder is the runnable CUB-200-2011 extension requested for Ours. It
contains the dataset contract, the required scratch guidance-teacher stage,
the Ours student entry point, a two-stage runner, and copy-ready H200 Issues.

## Locked first-run protocol

| Item | Value |
|---|---|
| Dataset | CUB-200-2011, official train/test split |
| Samples | train `5,994`, test `5,794`, total `11,788` |
| Classes | `200` |
| Teacher | scratch CIFAR-style ResNet56, input `32x32`, 300 epochs |
| Student | scratch DeiT-Ti, input `224x224`, 300 epochs |
| Ours base | researcher-sync, batch `64`, eval batch `200`, FP32, seed `1` |
| Optimizer | AdamW, `5e-4 -> 5e-6`, weight decay `0.05` |
| Warm-up | 20 epochs, factor `0.001` |
| Guidance | `beta=2.5`, `tau=-0.02`, window 50, larger-grid alignment |
| Augmentation | public LG strong augmentation |
| Checkpoint selection | best official-test Top-1, matching current repository convention |

Both teacher and student are trained without external pretraining. The
official CUB page warns that CUB images overlap with ImageNet, so using
ImageNet-pretrained weights could contaminate the test set. The run log and
checkpoints record `pretrained=False`.

The 32-pixel teacher is a deliberate compatibility choice: the existing Ours
module and all V2 comparisons use ResNet56 stages at `32/16/8`, with the
larger-grid policy producing `32/16/14`. This first CUB run measures whether
that low-resolution guidance transfers to fine-grained recognition; it should
not be silently replaced by a high-resolution teacher.

## Files

```text
cub200/
├── dataset.py        # official metadata parser, validation, verified download
├── train_teacher.py  # scratch 32x32 ResNet56 teacher
├── train.py          # fixed CUB-200 Ours wrapper
├── run_pipeline.py   # teacher -> Ours timing/full sequence
├── H200_ISSUE.md     # copy-ready request bodies
└── README.md
```

The downloader uses the official CaltechDATA archive and verifies MD5
`97eceeb196236b17998738112f37df78`. To use a pre-mounted copy, point
`--data-dir` either at `CUB_200_2011/` or its parent and add `--no-download`
to the teacher/pipeline command.

## Local structural checks

```bash
python -m unittest discover -s tests -p 'test_cub200_pipeline.py'
python methods/Ours/cub200/train.py --help
python methods/Ours/cub200/train_teacher.py --help
```

## H200 order

Run the combined two-epoch timing request first:

```bash
python methods/Ours/cub200/run_pipeline.py --timing-run \
  --num-workers 4 \
  --output-dir /app/output/cub200_ours_pipeline_timing_seed1
```

Submit the combined full request only if the last line reports
`[POD_LIMIT_CHECK] status=PASS`:

```bash
python methods/Ours/cub200/run_pipeline.py --full-run \
  --num-workers 4 \
  --output-dir /app/output/cub200_ours_pipeline_300ep_seed1
```

The teacher run writes a self-contained `manifest.json`; the student stage
loads and hash-verifies the selected teacher through that manifest. Results
are kept in separate `teacher/` and `Ours/cub200/` directories, with
`sequence_status.json` at the root.

See [`H200_ISSUE.md`](H200_ISSUE.md) for the complete request fields.
