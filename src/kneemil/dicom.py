"""Reading DICOM series and sorting slices in space.

Slices are ordered by their position along the normal of the image plane
(``ImagePositionPatient`` projected on ``ImageOrientationPatient``), never by
file name. ``InstanceNumber`` is only a fallback.
"""

import os

import numpy as np
import pandas as pd

HEADER_TAGS = ["InstanceNumber", "ImagePositionPatient", "ImageOrientationPatient",
               "PixelSpacing", "Rows", "Columns"]


def _to_floats(v, n):
    try:
        out = [float(x) for x in v]
        return out if len(out) == n else None
    except Exception:
        return None


def slice_position(ipp, iop):
    """Position of a slice along the normal of its image plane."""
    normal = np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float))
    return float(np.dot(np.asarray(ipp, float), normal))


def read_series_headers(path):
    """Read the headers of every slice of a series and sort them in space.

    Returns a DataFrame with one row per slice, sorted, and the columns
    ``file, key, inst, ps_r, ps_c, rows, cols, ts`` (``ts`` is the transfer syntax).
    """
    import pydicom

    rows = []
    for f in sorted(f for f in os.listdir(path) if f.endswith(".dcm")):
        try:
            ds = pydicom.dcmread(os.path.join(path, f), stop_before_pixels=True, specific_tags=HEADER_TAGS)
        except Exception:
            continue
        ipp = _to_floats(getattr(ds, "ImagePositionPatient", None) or [], 3)
        iop = _to_floats(getattr(ds, "ImageOrientationPatient", None) or [], 6)
        ps = _to_floats(getattr(ds, "PixelSpacing", None) or [], 2)
        try:
            inst = int(ds.InstanceNumber)
        except Exception:
            inst = None
        key = slice_position(ipp, iop) if ipp is not None and iop is not None else np.nan
        try:
            ts = str(ds.file_meta.TransferSyntaxUID)
        except Exception:
            ts = ""
        rows.append(dict(file=f, key=key, inst=inst,
                         ps_r=ps[0] if ps else np.nan, ps_c=ps[1] if ps else np.nan,
                         rows=int(getattr(ds, "Rows", 0) or 0), cols=int(getattr(ds, "Columns", 0) or 0), ts=ts))
    df = pd.DataFrame(rows, columns=["file", "key", "inst", "ps_r", "ps_c", "rows", "cols", "ts"])
    if len(df) == 0:
        return df
    if df["key"].notna().all() and df["key"].nunique() > 1:
        df = df.sort_values("key")
    elif df["inst"].notna().all():
        df = df.sort_values("inst")
    return df.reset_index(drop=True)


def read_pixels(fp):
    """Read one slice as a 2D float32 array (rescale slope and intercept applied).

    Compressed transfer syntaxes (JPEG Lossless, JPEG 2000) need the
    ``pylibjpeg`` decoders to be installed.
    """
    import pydicom

    ds = pydicom.dcmread(fp)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:  # rare: colour or multi-frame, keep the first plane
        arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr[0]
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
    return arr * slope + inter
