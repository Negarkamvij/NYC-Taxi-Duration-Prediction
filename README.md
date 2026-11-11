# NYC Yellow Taxi Trip-Duration Prediction

**Negar Kamvij · CS672 Project 2 · Fall 2026**

[Notebook](Taxi_Project_2.ipynb) | [Open in Colab](https://colab.research.google.com/github/Negarkamvij/NYC-Taxi-Duration-Prediction/blob/main/Taxi_Project_2.ipynb)

## Objective
Build a TensorFlow neural-network regression model that provides reliable and improved accuracy for NYC Yellow Taxi trip duration, using combined taxi and NYC weather data.

## Project contents
1. Setup
2. Data preprocessing and weather integration
3. Chronological 80/20 training/validation split and 0–1 feature scaling
4. TensorFlow MLP, linear regression with no hidden layers, and DNN with two hidden layers
5. MSE/MAE, SGD/Adam/RMSProp, learning rates 0.001/0.01; 100 epochs and batch size 32
6. Training/validation loss plots for every fit
7. Best-model selection, parameters, Keras predictions and loss review
8. Outlier sensitivity and overfitting
9. PyTorch DNN extra credit and framework comparison
10. Results and conclusion

The notebook is self-contained and follows the project objective. It contains no added research question or separate weather-ablation study. The data use a fixed-seed sample of January–November 2020 TLC trips and hourly NYC ERA5 weather through Open-Meteo, an alternative source permitted by the assignment. Recorded trip distance is an input to offline regression; live pre-trip use would need an estimated distance. Original files remain intact.

## Status
The project-format revision is published. The final experiment set is running; verified results and the completed notebook outputs will be added after training finishes. The current experiment CSV records actual progress and is not a final result table.
