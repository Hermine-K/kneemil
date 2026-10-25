# The problem

## Task

For every knee MRI study, predict the probability of 12 findings: ACL, MCL, medial and lateral
meniscus tears, medial, lateral and patellofemoral osteoarthritis, effusion, synovitis, Baker's
cyst, bone contusion and fracture. Several findings can coexist, so this is a **multi-label**
problem: 12 independent probabilities per study.

## Data

| | Studies | What is available |
|---|---|---|
| Gold training studies | 58 | images, report, 12 expert labels |
| Other training studies | 4,349 | images and free-text report only |
| Test | about 1,300 | images only |

- About 570 GB of DICOM files, 24,371 training series, 3 to 14 series per study (median 5).
- Each series comes with three descriptors: anatomical plane, fluid sensitivity, fat suppression.
- Series typically hold 20 to 45 slices (median 30), with a long tail up to 320.
- Pixel spacing ranges from 0.07 to 1.17 mm depending on the scanner.
- Reports come from international sites, in English, Spanish, Turkish, Greek and other languages.

## Metric

The score is the **macro ROC AUC**: one AUC per finding, averaged with equal weight.

!!! info "What AUC measures"
    Pick at random one knee with a given finding and one without. The AUC is the probability
    that the model gives the first a higher score. Only the **order** of the scores matters:
    no decision threshold is needed, and the metric does not depend on prevalence.

## Constraints

- **Code competition**: a Kaggle notebook is submitted and re-run on the hidden test set.
- **No internet** during scoring and a 9 hour limit.
- **No data may be sent to a third party**: reports cannot go to an external LLM API, so the
  LLM must run locally on Kaggle.
- Local hardware had no GPU: every computation ran on Kaggle (T4 GPUs, CPU sessions).
