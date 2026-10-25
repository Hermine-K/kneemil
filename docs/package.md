# Python package

The `kneemil` package gathers the reusable parts of the pipeline as tested functions. The
Kaggle notebooks remain the exact code behind the reported scores; the package is the same logic
organised for reuse.

## Installation

```bash
pip install -e .                    # core: DICOM, volumes, labels, metrics
pip install -e ".[dicom-codecs]"    # JPEG and JPEG 2000 DICOM decoders
pip install -e ".[model]"           # torch and timm, for the MIL model
pip install -e ".[dev]"             # pytest and ruff
```

## Modules

| Module | Content |
|---|---|
| `kneemil.constants` | the 12 findings, the canonical volume layout |
| `kneemil.paths` | locating competition data and notebook outputs on Kaggle |
| `kneemil.dicom` | reading headers and pixels, sorting slices in space |
| `kneemil.volume` | series selection, millimetre crop, canonical volume, 2.5D windows |
| `kneemil.labels` | LLM prompt, tolerant parser, soft targets, memory-safe batching |
| `kneemil.metrics` | macro AUC and bootstrap confidence interval |
| `kneemil.model` | MIL model with per-finding attention, augmentations, inference |

## Examples

Build the canonical volume of one study:

```python
import pandas as pd
from kneemil.volume import build_volume, window_centers, make_windows

series = pd.read_csv(f"{comp}/train_series.csv")
study = series[series.StudyInstanceUID == uid]
volume, mask, chosen = build_volume(comp, "train", uid, study, size=224)
windows = make_windows(volume, window_centers(mask))   # (K, 3, 224, 224)
```

Parse an LLM answer into soft targets:

```python
from kneemil.labels import parse_answer, to_soft

categories = parse_answer(model_output)   # 12 categories, or None
targets = to_soft(categories)             # 12 floats
```

Evaluate with a confidence interval:

```python
from kneemil.metrics import macro_auc, bootstrap_macro_auc

score, per_finding = macro_auc(y_true, y_pred)
mean, (low, high) = bootstrap_macro_auc(y_true, y_pred)
```

## Tests

```bash
pytest -q
```

The tests build a small synthetic DICOM dataset (no patient data) whose file names and instance
numbers are deliberately shuffled, and check that slices are sorted by position, that missing
slots stay empty, that windows never cross two series, that the parser handles the answer
formats met on Kaggle and that batching never exceeds its memory budget. Model tests run when
`torch` and `timm` are installed.
