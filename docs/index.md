# KneeMIL

**Knee MRI abnormality detection from weak, multilingual radiology report supervision.**

KneeMIL is a complete pipeline built for the
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
challenge (Kaggle, 2026): predict 12 abnormalities for every knee MRI study.

Only 58 of the 4,407 training studies carry expert labels. The 4,349 others only have their
free-text radiology report, written in several languages, and no report is available at test
time. The core problem is therefore **supervision**, not vision.

![Pipeline](assets/pipeline.png)

## In one paragraph

A local open-weights LLM reads every training report and grades each finding on five
certainty levels, which become soft targets. An image model then learns from these targets
alone: each study is reduced to a fixed 64-slice canonical volume, cut into 2.5D windows,
encoded by a shared ConvNeXt Tiny, and pooled by **one attention head per finding**, so that
each abnormality looks at its own slices. The 58 expert-labelled studies are never used for
training; they only measure.

## Key results

| Stage | Metric | Value |
|---|---|---|
| LLM labels | gold macro AUC (58 studies) | 0.839 |
| Image model | gold macro AUC | **0.907** (95 % CI 0.879 to 0.932) |
| Image model, 1 seed | public leaderboard | 0.899 |
| Ensemble of 2 seeds | public leaderboard | **0.906** |

The image model scores higher than the text labels it was trained on: noise in the labels is
scattered, while the visual signal is consistent.

## Where to go next

- [The problem](problem.md): data, metric and constraints.
- [Architecture](architecture.md): every stage, with the reasoning behind it.
- [Results](results.md): what worked, per finding.
- [Lessons and limitations](lessons.md): what went wrong and what remains uncertain.
