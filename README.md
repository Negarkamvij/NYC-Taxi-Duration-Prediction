# NYC Yellow Taxi Trip-Duration Prediction
**Negar Kamvij · CS672 Project 2**

[Notebook](Taxi_Project_2.ipynb) | [Open in Colab](https://colab.research.google.com/github/Negarkamvij/NYC-Taxi-Duration-Prediction/blob/main/Taxi_Project_2.ipynb)

I combine NYC Yellow Taxi and hourly NYC weather data and compare the required TensorFlow linear, MLP and DNN regression models. I include a matching PyTorch DNN for extra credit.

## Current revision
The notebook is the main source of truth. It now removes pickup month and days-since-year-start from the inputs, reports the chronological split and monthly counts, and implements all-row and at-most-180-minute metrics. It includes the full 36-run grid, baselines, distance-free and weather comparisons, three-seed checks, PyTorch, and a loss-plot appendix. The previous notebook/module and old-feature runs are retained locally in backups/ and outputs/old_runs/.

**Training is still running.** The final outputs and conclusion will replace the marked unrun section after the experiments are verified. This is not yet the final homework version.
