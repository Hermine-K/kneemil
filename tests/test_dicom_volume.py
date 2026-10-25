import numpy as np
import pandas as pd
import pytest

from kneemil.constants import N_SLICES, SLOTS
from kneemil.dicom import read_pixels, read_series_headers, slice_position
from kneemil.paths import find_comp_dir, series_path
from kneemil.volume import (
    build_volume,
    choose_series,
    crop_mm,
    make_windows,
    normalize_to_uint8,
    pick_indices,
    slot_bounds,
    window_centers,
)


def test_slice_position_follows_plane_normal():
    iop = [0, 1, 0, 0, 0, -1]  # sagittal: normal along x
    assert slice_position([5.0, 1.0, 2.0], iop) == pytest.approx(-5.0)  # normal = (-1, 0, 0)
    assert slice_position([0, 7, 3], iop) == pytest.approx(0.0)


def test_headers_sorted_by_position_not_by_file_or_instance(fake_comp):
    h = read_series_headers(series_path(fake_comp, "train", "1.2.0", "1.2.0.0"))
    assert len(h) == 30
    assert h["key"].is_monotonic_increasing
    values = [read_pixels(f"{series_path(fake_comp, 'train', '1.2.0', '1.2.0.0')}/{f}")[0, 0] for f in h["file"]]
    assert values == sorted(values)  # pixel value encodes the true index
    assert h["ps_r"].iloc[0] == pytest.approx(0.4)


def test_find_comp_dir(fake_comp):
    import os
    assert os.path.samefile(find_comp_dir(base=os.path.dirname(fake_comp)), fake_comp)
    with pytest.raises(FileNotFoundError):
        find_comp_dir(base="/nonexistent")


def test_choose_series_fills_slots_and_never_borrows_another_plane(fake_comp):
    ser = pd.read_csv(f"{fake_comp}/train_series.csv")
    full = choose_series(ser[ser.StudyInstanceUID == "1.2.0"])
    assert full == ["1.2.0.0", "1.2.0.1", "1.2.0.2", None, "1.2.0.3"]
    partial = choose_series(ser[ser.StudyInstanceUID == "1.2.2"])
    assert partial[2:] == [None, None, None]
    assert len(set(x for x in full if x)) == len([x for x in full if x])  # each series used once


@pytest.mark.parametrize("n,k", [(20, 18), (30, 12), (320, 8), (3, 12)])
def test_pick_indices_cover_the_stack(n, k):
    idx = pick_indices(n, k)
    assert len(idx) == k and idx.min() >= 0 and idx.max() <= n - 1
    assert np.all(np.diff(idx) >= 0)
    if n >= 20:
        assert idx[0] <= 0.1 * n and idx[-1] >= 0.9 * (n - 1)


def test_crop_is_physical_size():
    img = np.ones((400, 400), np.float32)
    small_pixels = crop_mm(img, 0.25, 0.25, 64)  # 400 px x 0.25 mm = 100 mm < 140 mm: padded
    assert small_pixels.shape == (64, 64)
    assert small_pixels[0, 0] == 0 and small_pixels[32, 32] == pytest.approx(1)
    large_pixels = crop_mm(img, 0.5, 0.5, 64)  # 200 mm field of view: fully inside
    assert large_pixels.min() == pytest.approx(1)


def test_normalize_to_uint8_range():
    stack = np.random.RandomState(0).normal(100, 20, (5, 32, 32)).astype(np.float32)
    out = normalize_to_uint8(stack)
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255


def test_build_volume_and_mask(fake_comp):
    ser = pd.read_csv(f"{fake_comp}/train_series.csv")
    vol, mask, chosen = build_volume(fake_comp, "train", "1.2.0", ser[ser.StudyInstanceUID == "1.2.0"], size=64)
    assert vol.shape == (N_SLICES, 64, 64) and vol.dtype == np.uint8
    b = slot_bounds()
    assert mask[b[0][0]:b[0][1]].all() and not mask[b[3][0]:b[3][1]].any()
    assert mask.sum() == N_SLICES - SLOTS[3][3]


def test_windows_never_cross_slots(fake_comp):
    mask = np.ones(N_SLICES, bool)
    centers = window_centers(mask)
    assert len(centers) == N_SLICES - 2 * len(SLOTS)
    starts = {a for a, _ in slot_bounds()}
    ends = {e - 1 for _, e in slot_bounds()}
    assert not (set(centers) & (starts | ends))
    vol = np.arange(N_SLICES, dtype=np.uint8)[:, None, None].repeat(4, 1).repeat(4, 2)
    w = make_windows(vol, centers)
    assert w.shape == (len(centers), 3, 4, 4)
    assert (w[:, 1, 0, 0] == centers).all()


def test_windows_respect_mask():
    mask = np.ones(N_SLICES, bool)
    mask[30:40] = False
    centers = window_centers(mask)
    for c in centers:
        assert mask[c - 1] and mask[c] and mask[c + 1]
    assert make_windows(np.zeros((N_SLICES, 4, 4), np.uint8), np.array([], int)).shape == (0, 3, 4, 4)
