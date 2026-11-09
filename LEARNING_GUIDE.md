# How I will learn and explain this project

This guide is a study aid. Rewrite the answers in your own words; do not memorize claims you cannot explain. The notebook is the main project document. `taxi_workflow.py` contains the implementation used by the notebook.

## 1. State the problem
A passenger starts a ride with a known destination area. The model estimates how many minutes the ride will take. The target is elapsed time between pickup and drop-off, calculated after converting the timestamps consistently.

Explain why a duration is a number, making this regression rather than classification. Explain why final fare and actual completed-trip distance are unavailable at the prediction moment.

## 2. Explain the data decision
All twelve 2020 Yellow Taxi files are audited. Original files stay unchanged. A reproducible uniform sample makes the required 100-epoch experiments practical. January–November are split roughly 80/20 in time; December is an additional final test.

Explain the difference between preserving the original rows and deciding which rows have usable training labels. Negative durations are not valid labels. A long ride is not automatically an error. Unknown location codes remain an unknown category.

## 3. Explain the time split
Training is practice on earlier rides. Validation is used to choose settings on later rides. The final test is reserved for a frozen choice.

The boundary is a whole hour so one hourly weather timestamp is not split across train and validation. A ride that starts in training but finishes in validation is excluded from the training view. Explain why its label would not have been known at the boundary.

## 4. Explain the baselines before the networks
The simplest prediction is the typical training duration. A stronger prediction uses the typical training duration for a pickup/destination pair with enough examples. A neural network should be compared with these meaningful reference points.

MAE is the average size of the prediction mistake in minutes. RMSE gives larger errors more influence. A low MAE with a much larger RMSE means that some large mistakes need investigation. Neither metric should be hidden.

## 5. Explain the first three models
- Linear Keras model: no hidden layer, additive relationships between the encoded inputs and the output.
- MLP: one hidden layer of 64 units; can learn nonlinear combinations.
- DNN: two hidden layers of 64 and 32 units; greater depth does not guarantee better validation results.

Each initial run uses MAE, Adam, learning rate 0.01, 100 epochs and batch size 32. Only architecture changes in this first comparison. The remaining grid tests the other required settings.

An epoch is one pass through the training data. A batch is the set of 32 examples used for a parameter update. Learning rate controls update size. Loss is the quantity the optimizer tries to reduce.

We train all 100 epochs and save the best validation-MAE checkpoint. The best checkpoint can come from an earlier epoch. Report its epoch and the final epoch's score; do not describe checkpoint selection as early stopping.

## 6. Explain the weather question carefully
Weather/no-weather comparison must use the same trips, split, architecture and settings. It measures whether weather improves this predictive experiment, not whether rain causes congestion.

The current source is ERA5 historical reanalysis. It reconstructs the past and does not prove that the same values were available to a live system at pickup. A production design needs a source with publication/availability times. A backward timestamp match alone does not solve that limitation.

Many rides share one weather hour. Hundreds of snowy rides can still represent only a few weather events. Report both trip counts and distinct weather hours/days.

## 7. Work still to do
1. Read and explain the initial data and model results.
2. Run the remaining required grid in Colab. Preserve the `outputs/` directory, preferably on Drive.
3. Inspect all loss curves and numerical failures; select by the same validation MAE.
4. Run the matched weather/no-weather comparison.
5. Run the PyTorch DNN and compare with its corresponding TensorFlow DNN.
6. Freeze the model choice, then evaluate December once and investigate errors.
7. Write your findings in your own words and prepare the presentation.

Do not call the project complete while these stages are still pending.
