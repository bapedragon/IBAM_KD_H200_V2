"""Compatibility imports for the official LG implementation.

New code should import from :mod:`methods.LG.official_lg`.  This module remains
so historical result readers and tests keep working.
"""

from methods.LG.official_lg import (  # noqa: F401
    OFFICIAL_LG_COMMIT,
    STUDENT_BLOCK_INDICES,
    STUDENT_CHANNELS,
    TEACHER_CHANNELS,
    LocalityGuidance,
)

__all__ = [
    "OFFICIAL_LG_COMMIT",
    "STUDENT_BLOCK_INDICES",
    "STUDENT_CHANNELS",
    "TEACHER_CHANNELS",
    "LocalityGuidance",
]
