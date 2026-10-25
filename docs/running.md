# Running the pipeline

Everything runs on Kaggle: the dataset (570 GB) is mounted there for free, and nothing has to
be downloaded. The notebooks are in [`notebooks/`](https://github.com/Hermine-K/kneemil/tree/main/notebooks)
and are the exact code that produced the reported scores.

## Order and settings

Create each notebook from the competition page (*Code*, then *New Notebook*, then
*File > Import Notebook*), so that the competition data is attached.

| # | Notebook | Accelerator | Internet | Inputs | Time |
|---|---|---|---|---|---|
| 01 | `01_series_inventory` | none | on | competition | 35 min |
| 02 | `02_llm_labels` | GPU T4 x2 | on | competition | 3 h |
| 03 | `03_canonical_volumes` | none | off | competition, 01 | 25 min |
| 04 | `04_training` | GPU | on | competition, 01, 02, 03 | 70 min |
| 04b | `04b_training_second_seed` | GPU | on | competition, 01, 02, 03, 04 | 70 min |
| 05 | `05_inference_ensemble` | none | **off** | competition, 01, 04, 04b | 2 to 5 h at scoring |

Run each with *Save Version*, then *Save & Run All*, so that its outputs are kept and can be
attached to the next notebooks (*Add Input*, *Your Work*). Submit notebook 05 with
*Submit to Competition*.

## Practical notes

- **Inputs are located automatically.** Kaggle mounts notebook outputs several levels deep;
  every notebook walks the input tree (skipping the competition data) to find what it needs and
  prints what it found.
- **Submissions must have internet disabled**, otherwise Kaggle refuses them.
- **Inference runs well on CPU.** It avoids GPU queues entirely; scoring stays well under the 9
  hour limit.
- **Training needs a GPU.** A guard stops the notebook immediately if none is attached.
- **Outputs of a failed run are not kept.** Long runs save checkpoints, but a version that ends
  in an error loses its output; notebook 02 can resume from a previous `llm_raw.csv` attached
  as input.
