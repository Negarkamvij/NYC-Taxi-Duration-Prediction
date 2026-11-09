# NYC Taxi Duration Prediction

**Negar Kamvij · CS672 Project 2 · Fall 2026**

[Open in Colab](https://colab.research.google.com/github/Negarkamvij/NYC-Taxi-Duration-Prediction/blob/main/Taxi_Project_2.ipynb) | [Notebook](Taxi_Project_2.ipynb) | [Learning guide](LEARNING_GUIDE.md)

## Question
Does weather improve ride-duration predictions for later NYC taxi trips, and where do predictions fail?

## Current status
**Modeling started; not a completed submission.** All twelve 2020 taxi files have been audited (24,649,092 source rows, no source-row deletion). A reproducible 50,000-trip Jan–Nov sample is split chronologically into approximately 80% training and 20% validation, with whole weather hours kept together and overlapping rides purged. A separate 10,000-trip December sample is reserved for a final test.

Three initial TensorFlow experiments are complete: a linear model, one-hidden-layer MLP and two-hidden-layer DNN, each trained for 100 epochs with MAE, Adam, learning rate 0.01 and batch size 32. The remaining grid, weather comparison, PyTorch taxi-data comparison and final test remain pending.

[Actual results and limitations](reports/First_Models.md) | [Experiment log](reports/experiment_results.csv) | [Full-year source audit](reports/source_audit_2020.csv)

## Why this design
The prediction is made at pickup with the destination assumed known. Actual drop-off time, final fare, tips, tolls and completed-trip distance are excluded from model inputs. The recorded final destination zone is a proxy for the intended zone; the source does not verify it was known at pickup. All original data remain available, and training eligibility is documented separately.

Simple baselines, train-only preprocessing, a weather/no-weather comparison, subgroup error counts and a reserved final period make conclusions assessable. More complex models are not assumed to be better. COVID-era changes and seasonality remain limitations.

Weather is **ERA5 retrospective reanalysis through Open-Meteo**, not a live weather feed or archived forecast. The current study cannot claim point-in-time production readiness. See [weather provenance](reports/weather_provenance_2020.json) and the notebook's source and timestamp notes.

## Run and learn
1. Open the notebook in Colab, or install `requirements.txt` locally.
2. Read sections 1–5 before training. `taxi_workflow.py` is the single implementation called by the notebook.
3. Run the first architecture comparison. Inspect baselines and all metrics; saved successful matching runs are reusable.
4. Enable `RUN_FULL_GRID` for the explicit 36-configuration study. Use persistent storage such as Drive for `outputs/`; otherwise runtime resets lose trained weights and histories.
5. Finish model selection, weather comparison and PyTorch extra credit before reporting the project complete. Enable the final-test cell only after freezing the TensorFlow model choice.

The raw data downloads require several hundred MB of disk space. The grid can take substantial compute. Each fit saves a history CSV, loss plot, checkpoint and result log. Numerical failures are logged honestly.

## Files
- `Taxi_Project_2.ipynb`: main notebook, explanations, actual saved outputs and deliberate controls for remaining experiments.
- `taxi_workflow.py`: typed, documented data and modeling functions.
- `LEARNING_GUIDE.md`: explanations and questions for presentation preparation.
- `tests/test_workflow.py`: checks for time leakage, missing-weather handling, train-only scaling and grid coverage.
- `reports/`: real source audits, data hashes, split coverage, experiment results and loss curves.
- `notebooks/January_Audit.ipynb`: earlier January-only audit, kept as historical work.
- `requirements-training-snapshot.txt`: direct package versions used for current training; not a complete environment lock.

`data/`, `outputs/` and trained models are excluded from Git. Project 1 is unchanged.

Sources: [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), [Open-Meteo ERA5 documentation](https://open-meteo.com/en/docs/historical-weather-api). The assignment allows other weather sources. Expanded taxi months were authorized by the student after clarification.
