# First modeling milestone: November 9, 2026

## Completed work
All twelve 2020 TLC source files were audited: **24,649,092 original rows**, with no source-row deletion. A uniform random-priority sample contains 50,000 eligible January–November trips and 10,000 December trips. Seed: 672. Sampling does not balance weather or months.

The whole-hour validation boundary is August 28, 2020, 01:00 New York time. Two training-view trips that crossed the boundary are purged from that view, not deleted from the source. There are 39,999 training trips, 9,999 validation trips and 10,000 reserved December test trips. Numerical preprocessing is fitted on training only. Unknown zone codes remain an unknown category.

Actual drop-off time, fare, tips, tolls and completed-trip distance are excluded from inputs. The final destination zone is used as a proxy for the intended destination known at pickup. TLC does not establish that an intended destination was actually recorded before departure; this is a stated retrospective modeling assumption.

## Initial validation results
All three network runs used MAE loss, Adam, learning rate 0.01, 100 epochs and batches of 32. Checkpoints were chosen by validation MAE, and final-epoch values remain available in the log. These are **3 of 36** planned grid configurations.

| Model | Validation MAE, minutes | Best checkpoint epoch | Final epoch MAE, minutes |
| --- | ---: | ---: | ---: |
| mlp | 6.243 | 6 | 7.141 |
| dnn | 6.253 | 12 | 7.161 |
| linear | 8.343 | 13 | 8.670 |

The overall training-median baseline has validation MAE **8.823 minutes**. The training-only route-median baseline has **7.955 minutes**. The initial networks must be interpreted against both, not only the weaker baseline.

The MLP and DNN are nearly tied in this one-seed initial comparison. A difference of about 0.01 minutes does not establish meaningful superiority. RMSE remains much larger than MAE, and R-squared remains low; large recorded-duration errors still need investigation. Do not hide these metrics or claim all data are correct.

## Weather and time limitations
The validation sample covers 202 distinct rain-only hours but no snow/mixed hours. The reserved December sample covers 17 snow-only and 8 mixed hours, representing only a few calendar days. These are candidate reanalysis classifications, not verified road conditions. Many taxi trips share one weather hour.

ERA5 via Open-Meteo reconstructs historical weather and is not a historical live-feed availability record. A backward measurement-time match does not fix hindsight. The final destination proxy, one city-area weather grid, COVID-related changes, season and route changes limit production and causal interpretations.

## Not completed or claimed
- Remaining 33 grid configurations, including MSE and the other optimizer/learning-rate combinations.
- Matched weather/no-weather training comparison.
- PyTorch taxi-data comparison. Its code was checked on synthetic data only; that is not an extra-credit result.
- December prediction metrics. The final test has not been evaluated.
- Final findings, model selection and presentation.

Every loss curve for a completed fit is saved in `figures/`, and each actual 100-epoch history is saved beside this report. No missing experiment has fabricated results.
