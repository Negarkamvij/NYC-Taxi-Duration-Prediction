# NYC Taxi Duration Prediction

**Negar Kamvij · CS672 Project 2 · Fall 2026**

[Open in Colab](https://colab.research.google.com/github/Negarkamvij/NYC-Taxi-Duration-Prediction/blob/main/Taxi_Project_2.ipynb) | [Notebook](Taxi_Project_2.ipynb)

## Research question
Does weather improve predictions of NYC Yellow Taxi ride duration, especially during rain or snow?

The prediction is made at ride pickup, assuming the destination zone is known. Actual drop-off time, final fare, tips, tolls, and completed-trip distance are not input features. Historical weather must be checked for time zone, measurement interval, and availability before a join is implemented.

## Current status
**Starter project, not a completed submission.** The notebook contains the project objective, assignment checklist, data download, record-preserving taxi audit, weather audit questions, and experiment design. Models, weather joins, and model results have not been implemented. Do not interpret this repository as a production system or an Uber implementation.

## Start
1. Open the notebook in Colab.
2. Install the initial audit packages if needed: `pip install -r requirements.txt` (or `%pip install pandas numpy pyarrow` in Colab).
3. Run the setup and taxi audit cells in order. They download the original January 2020 TLC file if missing. The full file requires disk space and memory.
4. Select and document a weather source, then add the weather coverage audit and join.

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
- `requirements.txt`: initial data-audit dependencies only. Training dependencies will be added and tested later.
- `.gitignore`: excludes raw data, models, credentials, and generated results.

Project 1 remains in its separate repository. Source: [NYC TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Assignment weather example: [Meteostat, January 2020](https://meteostat.net/en/place/3BBKPQ?t=2020-01-01/2020-01-31).
