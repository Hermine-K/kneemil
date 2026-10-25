# Lessons and limitations

## What went wrong, and what it taught

| Problem | Cause | Fix |
|---|---|---|
| 19 of 58 LLM answers lost | strict parsing: one unexpected word rejected a whole answer | tolerant parser, synonyms, finding words read as present |
| Clear ACL tear labelled negative | Spanish abbreviation LCA unknown to the prompt | multilingual glossary |
| CUDA out of memory after 3 hours | fixed batch size, reports sorted by length | batches sized by token count, automatic split |
| Notebook could not find its inputs | Kaggle mounts outputs several levels deep | walk the input tree |
| Training crashed at start | session launched without a GPU | explicit GPU guard with a clear message |
| Submission refused | notebook saved with internet enabled | internet off for the inference notebook |
| GPU queue of several hours | shared Kaggle resources | inference moved to CPU |
| Ensemble looked worse on gold | 58 studies too few to judge a 0.004 difference | keep both, let the larger test set decide |

## Limitations

- **Small gold set.** 58 studies, with only 9 to 35 positives per finding: per-finding AUCs are
  unstable and the bootstrap intervals are wide.
- **The gold set is not perfectly untouched.** It was used to tune the labelling prompt and then
  to evaluate the image model (never to train it).
- **Text has a ceiling.** Expert labels seem image-based; synovitis and effusion are often absent
  from reports.
- **Prevalence shift.** Label prevalence may differ between training, public and private test sets.
- **Resolution.** Volumes are stored at 224 x 224 pixels to fit Kaggle's 20 GB output limit;
  higher resolution could help small structures.

## What could come next

- Self-training: use the image model's predictions to refine the LLM targets, then retrain.
- More diverse ensembles (different backbones or resolutions) rather than different seeds.
- Inspecting the attention weights per finding to check that the model looks at the right slices.
- Transfer to digital pathology, where the same structure applies: one label per slide, tiles as
  instances, pathology reports as weak supervision.
