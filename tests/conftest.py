"""Synthetic DICOM data shaped like the competition layout (no real patient data)."""

import os
import random

import numpy as np
import pandas as pd
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402

ORIENTATIONS = {"Sagittal": [0, 1, 0, 0, 0, -1], "Coronal": [1, 0, 0, 0, 0, -1], "Axial": [1, 0, 0, 0, 1, 0]}


def write_slice(path, i, plane, n, spacing, rows=320, cols=300):
    """One synthetic slice whose pixel value encodes its true spatial index ``i``."""
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = Dataset()
    ds.file_meta = meta
    iop = ORIENTATIONS[plane]
    normal = np.cross(iop[:3], iop[3:])
    ds.ImageOrientationPatient = iop
    ds.ImagePositionPatient = [float(x) for x in normal * i * 3.0]
    ds.InstanceNumber = n - i  # deliberately reversed: sorting must use the position
    ds.PixelSpacing = [spacing, spacing]
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = np.full((rows, cols), i * 10, np.uint16).tobytes()
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    pydicom.dcmwrite(path, ds, enforce_file_format=True)


@pytest.fixture(scope="session")
def fake_comp(tmp_path_factory):
    """A tiny competition folder: 3 studies, the last one with only 2 series."""
    root = tmp_path_factory.mktemp("comp")
    rng = random.Random(0)
    rows = []
    specs_full = [("Sagittal", 1, 1, 30), ("Sagittal", 0, 0, 25), ("Coronal", 1, 1, 20), ("Axial", 1, 0, 22)]
    for s in range(3):
        study = f"1.2.{s}"
        specs = specs_full if s < 2 else specs_full[:2]
        for j, (plane, fluid, fat, n) in enumerate(specs):
            series = f"{study}.{j}"
            d = os.path.join(root, "train_series", study, series)
            os.makedirs(d)
            order = list(range(n))
            rng.shuffle(order)  # file names in random order
            for i in order:
                write_slice(os.path.join(d, f"{generate_uid()}.dcm"), i, plane, n, 0.4)
            rows.append(dict(StudyInstanceUID=study, SeriesInstanceUID=series, Fluid_Sensitive=fluid,
                             Fat_Suppression=fat, Anatomical_Plane=plane))
    pd.DataFrame(rows).to_csv(os.path.join(root, "train_series.csv"), index=False)
    pd.DataFrame({"StudyInstanceUID": ["1.2.0"]}).to_csv(os.path.join(root, "test.csv"), index=False)
    return str(root)
