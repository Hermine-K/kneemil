"""KneeMIL: knee MRI abnormality detection from weak, multilingual report supervision.

Modules
-------
constants  the 12 findings and the canonical volume layout
paths      locating competition data and notebook outputs on Kaggle
dicom      reading and spatially sorting DICOM series
volume     series selection, canonical volume, 2.5D windows
labels     LLM report labelling: prompt, tolerant parser, soft targets, batching
metrics    macro AUC and bootstrap confidence interval
model      MIL model with per-finding attention (requires torch and timm)
"""

__version__ = "1.0.0"

from .constants import N_SLICES, SLOTS, TARGETS  # noqa: F401
