# NYC Yellow Taxi Trip-Duration Prediction
**Negar Kamvij · CS672 Project 2 · Fall 2026**

[Notebook](Taxi_Project_2.ipynb) | [Open in Colab](https://colab.research.google.com/github/Negarkamvij/NYC-Taxi-Duration-Prediction/blob/main/Taxi_Project_2.ipynb)

## Objective
I combine NYC Yellow Taxi records and NYC weather data to build and compare TensorFlow neural-network regression models for trip duration. I include a matching PyTorch DNN for extra credit.

## Completed work
- Chronological 80/20 training/validation split, with whole weather hours kept together and crossing training labels excluded.
- Train-only imputation and independent 0–1 scaling. Pickup month and year-progress inputs are removed; monthly split counts remain visible.
- All 36 required TensorFlow configurations: linear, MLP and DNN; MSE/MAE; SGD/Adam/RMSProp; two learning rates; 100 epochs and batch size 32.
- Training/validation loss plots for every fit, with the required plots in the notebook appendix.
- Training-median and distance-only Keras baselines, all-row and ≤180-minute metrics, saved predictions and the 20 largest errors.
- Distance/hour/rain error analysis, distance-free and weather comparisons, and three seeds for the top three configurations.
- A real PyTorch taxi-data DNN run, matching TensorFlow reference and framework comparison.
- One additional dropout DNN and a conclusion based only on executed results.

The main notebook is self-contained and is the source of truth. The `.py` file contains its matching functions. The uploaded Colab run has saved outputs with no Python error outputs. The saved notebook includes the executed subgroup tables for trips up to 180 minutes, the long-trip frequency comparison, and the refreshed residual plot. In total, 47 Keras fits and one PyTorch fit completed 100 epochs each; this includes the required grid and additional comparisons. No runs failed.

## Main results
The selected TensorFlow model is the MLP with MAE loss, SGD and learning rate 0.01. In the uploaded Colab run, its all-row validation MAE is 5.004 minutes and its three-seed configuration mean is 5.013 ± 0.010 minutes. Its selected checkpoint is epoch 40. Long recorded durations strongly affect RMSE. All-row metrics remain primary; filtered metrics do not replace them.

[Conclusion from the uploaded Colab run](reports/Conclusion.md)

The following CSV reports are retained from the earlier local run and differ from the latest Colab outputs in the notebook: [Grid results](reports/grid_results.csv) | [Seed comparison](reports/seed_summary.csv) | [Framework comparison](reports/framework_comparison.csv)

## Running
I run the notebook in order. In Colab, Setup mounts Google Drive and uses `MyDrive/taxi_project` for persistent files; locally it uses the working directory. Put `assignment_sample.parquet` in `outputs/` to reuse the same sample and avoid downloading eleven taxi months. The submission package includes that file. If no sample is present, the notebook downloads the sources and creates it.

`requirements-tested.txt` records the tested local package versions. The retained CSV reports contain earlier local CPU measurements. The notebook and conclusion use the uploaded Colab results, including measured framework times of 69.3 seconds for TensorFlow and 151.9 seconds for PyTorch. Matching caches are reused only when configuration, feature list, input width and row context agree. Old-feature runs remain in the local `outputs/old_runs` folder and are never searched for current results.

Raw files, weights, backups and run caches are excluded from Git. The submission package contains the current sample and models. Prior files are retained locally in `backups/`.

## Data and limits
I use a fixed-seed sample of January–November 2020 TLC trips and ERA5 weather through Open-Meteo, an alternative source allowed by the assignment. The original files and long trips are kept. Recorded distance is an offline input; pre-trip use would need estimated distance. ERA5 is retrospective weather, not a live forecast. The train/validation period crosses COVID-related travel changes, and validation was used for both selection and reporting. No test set was added.
