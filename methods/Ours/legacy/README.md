# Archived Ours integration before researcher-code synchronization

This folder records the Ours behavior used by the completed historical runs
before the researcher screenshots shared on 2026-07-21 were synchronized.

The complete executable snapshot is preserved by Git commit `ee2dc55`:

```bash
git show ee2dc55:methods/Ours/core.py
git show ee2dc55:methods/Ours/ours.py
git show ee2dc55:methods/Ours/chaoyang/train.py
```

The two files in this folder extract the parts that materially differed from
the researcher implementation:

- `pre_researcher_controller.py`: controller used by the earlier runs. It
  observed alignment loss only, initialized the first derivative to zero, and
  required a decreasing regime before allowing the permanent stop.
- `pre_researcher_chaoyang_profile.py`: the previous Chaoyang wrapper values,
  including batch 128 and the paper-wording `32/16/8` grid.

Do not use these archived settings for the new researcher-synchronized run.
They remain only so the older result files can be traced to their exact code.
