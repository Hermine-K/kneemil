"""Locating the competition data and the outputs of other Kaggle notebooks.

Kaggle mounts notebook outputs several levels deep (for example
``/kaggle/input/notebooks/<user>/<notebook>/``). Files are therefore found by
walking the input tree, while never entering the competition data itself
(820,000 DICOM files).
"""

import glob
import os

KAGGLE_INPUT = "/kaggle/input"
COMPETITION = "rsna-knee-abnormality-detection"


def find_comp_dir(base=KAGGLE_INPUT):
    """Return the mounted competition folder (the one containing ``test.csv``)."""
    for c in (os.path.join(base, "competitions", COMPETITION), os.path.join(base, COMPETITION)):
        if os.path.exists(os.path.join(c, "test.csv")) or os.path.exists(os.path.join(c, "train.csv")):
            return c
    for pattern in (f"{base}/*/test.csv", f"{base}/*/*/test.csv"):
        hits = glob.glob(pattern)
        if hits:
            return os.path.dirname(hits[0])
    raise FileNotFoundError("Competition data not found: add it as an input of the notebook.")


def walk_inputs(base=KAGGLE_INPUT, max_depth=5):
    """``os.walk`` over attached inputs, skipping the competition data."""
    for root, dirs, files in os.walk(base):
        depth = root[len(base):].count(os.sep)
        dirs[:] = [d for d in dirs if d != "competitions" and "rsna-knee" not in d
                   and not d.endswith("_series") and depth < max_depth]
        yield root, dirs, files


def find_input(name, base=KAGGLE_INPUT, working="/kaggle/working"):
    """Path of a file produced by another notebook attached as input, or None."""
    for root, _dirs, files in walk_inputs(base):
        if name in files:
            return os.path.join(root, name)
    p = os.path.join(working, name)
    return p if os.path.exists(p) else None


def series_csv_path(comp, split):
    """``train_series.csv`` or ``test_series.csv``."""
    return os.path.join(comp, f"{split}_series.csv")


def series_path(comp, split, study, series):
    """Folder holding the DICOM slices of one series."""
    return os.path.join(comp, f"{split}_series", str(study), str(series))
