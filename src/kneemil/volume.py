"""Canonical volume and 2.5D windows.

A study is turned into a fixed 64-slice volume made of 5 slots (see
:data:`kneemil.constants.SLOTS`). Each slot is filled from one series chosen
with the descriptors provided by the competition (plane, fluid sensitivity,
fat suppression). Missing slots stay empty and are flagged in a validity mask;
they are never filled with a series from another plane.
"""

import os

import numpy as np

from .constants import CROP_MM, N_SLICES, SAMPLE_HI, SAMPLE_LO, SLOTS
from .dicom import read_pixels, read_series_headers
from .paths import series_path


def choose_series(study_series):
    """Pick one series per slot.

    Parameters
    ----------
    study_series : pandas.DataFrame
        Rows of the series CSV for one study, with the columns
        ``SeriesInstanceUID, Anatomical_Plane, Fluid_Sensitive, Fat_Suppression``
        and optionally ``n`` (number of slices).

    Returns
    -------
    list
        Five series UIDs, ``None`` for an empty slot.
    """
    df = study_series.copy()
    if "n" not in df.columns:
        df["n"] = 0
    df = df[df["n"].fillna(0) >= 3] if (df["n"] > 0).any() else df
    used, out = set(), []
    for _, plane, kind, _k in SLOTS:
        cand = df[(df["Anatomical_Plane"] == plane) & (~df["SeriesInstanceUID"].isin(used))]
        if kind == "fluid":
            cand = cand[cand["Fluid_Sensitive"] == 1]
            cand = cand.assign(score=cand["Fat_Suppression"] * 1000 + cand["n"].clip(upper=200))
        elif kind == "anat":
            cand = cand[cand["Fat_Suppression"] == 0]
            cand = cand.assign(score=(1 - cand["Fluid_Sensitive"]) * 1000 + cand["n"].clip(upper=200))
        else:
            cand = cand.assign(score=cand["Fluid_Sensitive"] * cand["Fat_Suppression"] * 1000
                               + cand["n"].clip(upper=200))
        if len(cand) == 0:
            out.append(None)
            continue
        best = cand.sort_values("score", ascending=False).iloc[0]["SeriesInstanceUID"]
        used.add(best)
        out.append(best)
    return out


def pick_indices(n, k, lo=SAMPLE_LO, hi=SAMPLE_HI):
    """``k`` evenly spaced slice indices between ``lo`` and ``hi`` of an ``n``-slice stack."""
    pos = np.linspace(lo, hi, k)
    return np.clip(np.round(pos * (n - 1)).astype(int), 0, n - 1)


def crop_mm(img, ps_r, ps_c, size, crop=CROP_MM):
    """Square of ``crop`` millimetres around the image centre, resized to ``size`` x ``size``.

    The crop is zero-padded when the field of view is smaller. Cropping in
    millimetres rather than pixels makes the anatomy scale independent of the
    scanner resolution.
    """
    import cv2

    if not (ps_r and ps_c) or np.isnan(ps_r) or np.isnan(ps_c) or ps_r <= 0 or ps_c <= 0:
        ps_r = ps_c = crop / max(img.shape)  # unknown spacing: keep the whole image
    hr = int(round(crop / 2.0 / ps_r))
    hc = int(round(crop / 2.0 / ps_c))
    H, W = img.shape
    cy, cx = H // 2, W // 2
    canvas = np.zeros((2 * hr, 2 * hc), dtype=np.float32)
    y0, y1 = max(0, cy - hr), min(H, cy + hr)
    x0, x1 = max(0, cx - hc), min(W, cx + hc)
    oy, ox = y0 - (cy - hr), x0 - (cx - hc)
    canvas[oy:oy + (y1 - y0), ox:ox + (x1 - x0)] = img[y0:y1, x0:x1]
    interp = cv2.INTER_AREA if canvas.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(canvas, (size, size), interpolation=interp)


def normalize_to_uint8(stack):
    """Per-series intensity normalisation: clip to percentiles 1 and 99, scale to 0..255."""
    lo, hi = np.percentile(stack, [1, 99])
    stack = np.clip((stack - lo) / max(hi - lo, 1e-6), 0, 1)
    return (stack * 255).astype(np.uint8)


def build_volume(comp, split, study, study_series, size=224, sorted_files=None):
    """Build the canonical volume of one study.

    Parameters
    ----------
    comp : str
        Competition data folder.
    split : str
        ``"train"`` or ``"test"``.
    study : str
        StudyInstanceUID.
    study_series : pandas.DataFrame
        Series descriptors of this study (see :func:`choose_series`).
    size : int
        Output image size in pixels.
    sorted_files : dict, optional
        ``{series: (sorted file list, ps_r, ps_c)}`` from a previous inventory;
        headers are read again when missing.

    Returns
    -------
    volume : numpy.ndarray
        ``(64, size, size)`` uint8.
    mask : numpy.ndarray
        ``(64,)`` bool, True for real slices.
    chosen : list
        Series UID chosen for each slot.
    """
    vol = np.zeros((N_SLICES, size, size), dtype=np.uint8)
    mask = np.zeros(N_SLICES, dtype=bool)
    chosen = choose_series(study_series)
    start = 0
    for (_name, _plane, _kind, k), sid in zip(SLOTS, chosen):
        if sid is not None:
            try:
                p = series_path(comp, split, study, sid)
                if sorted_files is not None and sid in sorted_files:
                    files, ps_r, ps_c = sorted_files[sid]
                else:
                    h = read_series_headers(p)
                    files = h["file"].tolist()
                    ps_r = float(h["ps_r"].median()) if len(h) else np.nan
                    ps_c = float(h["ps_c"].median()) if len(h) else np.nan
                if len(files) > 0:
                    cache, sl = {}, []
                    for i in pick_indices(len(files), k):
                        if i not in cache:
                            cache[i] = crop_mm(read_pixels(os.path.join(p, files[i])), ps_r, ps_c, size)
                        sl.append(cache[i])
                    vol[start:start + k] = normalize_to_uint8(np.stack(sl))
                    mask[start:start + k] = True
            except Exception as e:  # one broken series must not lose the whole study
                print(f"[volume] {study} {sid}: {type(e).__name__} {e}")
        start += k
    return vol, mask, chosen


def slot_bounds():
    """``[(start, end), ...]`` slice ranges of the 5 slots."""
    bounds, s = [], 0
    for _, _, _, k in SLOTS:
        bounds.append((s, s + k))
        s += k
    return bounds


def window_centers(mask):
    """Centres ``c`` such that slices ``c-1, c, c+1`` are valid and in the same slot.

    A 2.5D window never mixes two different series.
    """
    centers = []
    for a, b in slot_bounds():
        for c in range(a + 1, b - 1):
            if mask[c - 1] and mask[c] and mask[c + 1]:
                centers.append(c)
    return np.array(centers, dtype=int)


def make_windows(vol, centers):
    """Stack 2.5D windows: three neighbouring slices used as the three channels, ``(K, 3, H, W)``."""
    if len(centers) == 0:
        return np.zeros((0, 3) + vol.shape[1:], vol.dtype)
    return np.stack([vol[c - 1:c + 2] for c in centers])
