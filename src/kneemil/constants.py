"""Findings and canonical volume layout."""

#: The 12 findings predicted for every study, in submission order.
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

#: Canonical volume slots: (name, plane, sequence filter, number of slices).
#: The order never changes, so a slice index always means the same plane and contrast.
SLOTS = [
    ("sag_fluid", "Sagittal", "fluid", 18),
    ("sag_anat", "Sagittal", "anat", 14),
    ("cor_fluid", "Coronal", "fluid", 12),
    ("cor_anat", "Coronal", "anat", 8),
    ("axial", "Axial", "any", 12),
]

#: Total number of slices in a canonical volume (64).
N_SLICES = sum(s[3] for s in SLOTS)

#: Side of the square crop, in millimetres, taken around the image centre.
CROP_MM = 140.0

#: Slices are sampled between these fractions of each stack, so that
#: peripheral structures (collateral ligaments, lateral meniscus) are kept.
SAMPLE_LO, SAMPLE_HI = 0.06, 0.94
