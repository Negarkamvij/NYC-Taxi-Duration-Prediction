# NYC Taxi Duration Prediction

**Negar Kamvij · CS672 Project 2 · Fall 2026**

[Open in Colab](https://colab.research.google.com/github/Negarkamvij/NYC-Taxi-Duration-Prediction/blob/main/Taxi_Project_2.ipynb) | [Notebook](Taxi_Project_2.ipynb)

## Research question
Does weather improve predictions of NYC Yellow Taxi ride duration, especially during rain or snow?

The prediction is made at ride pickup, assuming the destination zone is known. Actual drop-off time, final fare, tips, tolls, and completed-trip distance are not input features. Historical weather must be checked for time zone, measurement interval, and availability before a join is implemented.

## Current status
**Data audit stage; not a completed submission.** All six current code cells have been executed with saved outputs. The notebook now includes the full January taxi audit, candidate hourly ERA5 weather coverage, time-zone checks, a retrospective timestamp association, and a preliminary split audit. Models are not trained and no model accuracy is claimed.

### What the audit found
All 6,405,008 taxi source rows are retained. The weather candidate has 744 January hours, 67 with positive rain and 17 with positive snowfall. The proposed validation period has only four rain-only hours and no snowy hours. This is too limited for strong weather-specific generalization claims. Many taxi rows share the same weather and are not independent weather events.

**Next decision:** review a broader period and weather source before finalizing the training split. ERA5 is reanalysis, not an archived live observation or forecast. Its current use is a retrospective audit, not evidence of production-time availability.

[Read the audit findings](reports/January_Audit.md) | [Data hashes and environment](reports/data_manifest.json)

## Start
1. Open the notebook in Colab.
2. Install the initial audit packages if needed: `pip install -r requirements.txt` (or `%pip install pandas numpy pyarrow matplotlib` in Colab).
3. Run the setup and taxi audit cells in order. They download the original January 2020 TLC file if missing. The full file requires disk space and memory.
4. Read the weather and split audit before model training. The weather cell downloads a small ERA5 candidate file if missing. A final source and period still need to be selected.

January 2020 is the initial candidate because the assignment gives a January weather example. It is not presented as a mandatory assignment month. The final period may change based on weather coverage.

## Required experiments, still planned
- TensorFlow/Keras linear model with no hidden layers, MLP, and DNN with at least two hidden layers.
- Chronological 80/20 training/validation split with leakage checks.
- MSE and MAE losses; SGD, Adam, and RMSProp; multiple learning rates.
- 100 epochs per fit, batch size 32, loss curves for every fit.
- Common validation metrics, selected-model parameters, predictions, and error review.
- PyTorch DNN with at least two hidden layers for extra credit.

## Additional investigation
Weather/no-weather comparison, errors by weather and time, and examples of large mistakes. Rain and snow are hypotheses, not assumed findings. Sparse snow coverage will be reported as a limitation.

## Repository contents
- `Taxi_Project_2.ipynb`: starter notebook with simple English explanations and typed, documented functions.
- `requirements.txt`: taxi/weather-audit dependencies only. Training dependencies will be added and tested later.
- `requirements-audit.lock.txt`: direct package versions used for the executed audit.
- `reports/`: actual audit findings, source metadata, and file hashes; raw data remains excluded.
- `.gitignore`: excludes raw data, models, credentials, and generated results.

Project 1 remains in its separate repository. Source: [NYC TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Assignment weather example: [Meteostat, January 2020](https://meteostat.net/en/place/3BBKPQ?t=2020-01-01/2020-01-31).
