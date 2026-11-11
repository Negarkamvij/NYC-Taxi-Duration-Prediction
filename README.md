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

The main notebook is self-contained and is the source of truth. The `.py` file contains its matching functions. All 18 code cells passed end to end. In total, 47 Keras fits and one PyTorch fit completed 100 epochs each; this includes the required grid and additional comparisons. No runs failed.

## Main results
The selected TensorFlow model is the MLP with MAE loss, SGD and learning rate 0.01. Its all-row validation MAE is 5.010 minutes; the three-seed configuration mean is 5.014 ± 0.004 minutes. Long recorded durations strongly affect RMSE. All-row metrics remain primary; filtered metrics do not replace them.

[Conclusion](reports/Conclusion.md) | [Grid results](reports/grid_results.csv) | [Seed comparison](reports/seed_summary.csv) | [Framework comparison](reports/framework_comparison.csv)

## Running
I run the notebook in order. In Colab, Setup mounts Google Drive and uses `MyDrive/taxi_project` for persistent files; locally it uses the working directory. Put `assignment_sample.parquet` in `outputs/` to reuse the same sample and avoid downloading eleven taxi months. The submission package includes that file. If no sample is present, the notebook downloads the sources and creates it.

`requirements-tested.txt` records the tested local package versions. Saved reference timings are local-CPU measurements, not estimates for Colab. Matching caches are reused only when configuration, feature list, input width and row context agree. Old-feature runs remain in the local `outputs/old_runs` folder and are never searched for current results.

Raw files, weights, backups and run caches are excluded from Git. The submission package contains the current sample and models. Prior files are retained locally in `backups/`.

## Data and limits
I use a fixed-seed sample of January–November 2020 TLC trips and ERA5 weather through Open-Meteo, an alternative source allowed by the assignment. The original files and long trips are kept. Recorded distance is an offline input; pre-trip use would need estimated distance. ERA5 is retrospective weather, not a live forecast. The train/validation period crosses COVID-related travel changes, and validation was used for both selection and reporting. No test set was added.
