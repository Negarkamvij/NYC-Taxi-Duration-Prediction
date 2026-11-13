from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
import json
import shutil
import time
import numpy as np
import pandas as pd

SEED = 672
WEATHER = ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "snowfall", "wind_speed_10m", "surface_pressure"]
NUMERIC = ["passenger_count", "log_trip_distance"]
CATEGORICAL = ["pickup_zone", "destination_zone", "pickup_hour", "pickup_weekday"]
TARGET = "duration_minutes"


def download_file(url: str, destination: Path) -> Path:
    """Download a missing source file without overwriting an existing file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        partial = Path(str(destination) + ".part")
        with urlopen(Request(url, headers={"User-Agent": "CS672-project"}), timeout=120) as response:
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output)
        partial.replace(destination)
    return destination


def weather_classes(frame: pd.DataFrame) -> pd.Series:
    """Label precipitation, preserving missing values as unknown."""
    missing = frame[["rain", "snowfall", "precipitation"]].isna().any(axis=1)
    rain = frame["rain"].gt(0)
    snow = frame["snowfall"].gt(0)
    labels = np.select([missing, rain & snow, snow, rain, frame["precipitation"].eq(0)],
                       ["unknown", "mixed", "snow", "rain", "dry"], default="other_or_trace")
    return pd.Series(labels, index=frame.index)


def load_weather(root: Path) -> pd.DataFrame:
    """Load hourly NYC ERA5 values in Celsius and UTC."""
    parameters = {
        "latitude": 40.7075, "longitude": -74.0113,
        "start_date": "2019-12-31", "end_date": "2021-01-02",
        "hourly": ",".join(WEATHER), "models": "era5", "timezone": "UTC",
        "temperature_unit": "celsius", "wind_speed_unit": "kmh", "precipitation_unit": "mm",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(parameters)
    path = download_file(url, root / "data/weather_era5_2020.json")
    payload = json.loads(path.read_text())
    expected_units = {"temperature_2m": "°C", "relative_humidity_2m": "%", "precipitation": "mm",
                      "rain": "mm", "snowfall": "cm", "wind_speed_10m": "km/h", "surface_pressure": "hPa"}
    assert all(payload["hourly_units"][name] == unit for name, unit in expected_units.items())
    weather = pd.DataFrame(payload["hourly"])
    weather["weather_time_utc"] = pd.to_datetime(weather.pop("time"), utc=True)
    weather = weather.sort_values("weather_time_utc").reset_index(drop=True)
    assert weather["weather_time_utc"].is_unique
    assert weather["weather_time_utc"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    weather["weather_class"] = weather_classes(weather)
    return weather


def local_utc(values: pd.Series) -> pd.Series:
    """Convert New York wall-clock times, marking unresolved DST times unavailable."""
    parsed = pd.to_datetime(values, errors="coerce")
    localized = parsed.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    return localized.dt.tz_convert("UTC")


def prepare_sample(root: Path, weather: pd.DataFrame, sample_size: int = 50000) -> pd.DataFrame:
    """Uniformly sample eligible January-November rows while preserving raw files."""
    sample_pool = pd.DataFrame()
    monthly_counts = []
    columns = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID", "passenger_count", "trip_distance"]
    for month in range(1, 12):
        filename = f"yellow_tripdata_2020-{month:02d}.parquet"
        path = download_file("https://d37ci6vzurychx.cloudfront.net/trip-data/" + filename, root / "data" / filename)
        raw = pd.read_parquet(path, columns=columns)
        pickup = local_utc(raw["tpep_pickup_datetime"])
        dropoff = local_utc(raw["tpep_dropoff_datetime"])
        duration = (dropoff - pickup).dt.total_seconds() / 60
        local = pickup.dt.tz_convert("America/New_York")
        intended_month = local.dt.year.eq(2020) & local.dt.month.eq(month)
        eligible = pickup.notna() & dropoff.notna() & np.isfinite(duration) & duration.gt(0) & intended_month
        monthly_counts.append({"month": month, "source_rows": len(raw), "eligible_rows": int(eligible.sum()),
                               "invalid_or_outside_month": int((~eligible).sum()), "duration_over_180_minutes": int(duration.gt(180).sum())})
        priorities = np.random.default_rng(SEED + month).random(len(raw))
        positions = np.flatnonzero(eligible.to_numpy())
        if len(positions) > sample_size:
            positions = positions[np.argpartition(priorities[positions], sample_size - 1)[:sample_size]]
        part = pd.DataFrame({
            "row_id": [f"{month:02d}:{position}" for position in positions],
            "pickup_utc": pickup.iloc[positions].to_numpy(),
            "dropoff_utc": dropoff.iloc[positions].to_numpy(),
            TARGET: duration.iloc[positions].to_numpy(),
            "priority": priorities[positions],
            "passenger_count": raw["passenger_count"].iloc[positions].to_numpy(),
            "trip_distance": raw["trip_distance"].iloc[positions].to_numpy(),
        })
        for original, feature in [("PULocationID", "pickup_zone"), ("DOLocationID", "destination_zone")]:
            zones = raw[original].iloc[positions]
            zones = zones.where(zones.between(1, 263) & zones.mod(1).eq(0), 0)
            part[feature] = zones.fillna(0).astype(int).astype(str).to_numpy()
        part["passenger_count"] = part["passenger_count"].where(part["passenger_count"].ge(0))
        valid_distance = part["trip_distance"].where(part["trip_distance"].ge(0) & np.isfinite(part["trip_distance"]))
        part["log_trip_distance"] = np.log1p(valid_distance)
        sample_pool = pd.concat([sample_pool, part], ignore_index=True).nsmallest(sample_size, "priority")
    sample = sample_pool.sort_values("pickup_utc").reset_index(drop=True)
    sample["pickup_utc"] = pd.to_datetime(sample["pickup_utc"], utc=True)
    sample["dropoff_utc"] = pd.to_datetime(sample["dropoff_utc"], utc=True)
    sample["weather_time_utc"] = sample["pickup_utc"].dt.floor("h")
    sample = sample.merge(weather, on="weather_time_utc", how="left", validate="many_to_one")
    local = sample["pickup_utc"].dt.tz_convert("America/New_York")
    sample["pickup_hour"] = local.dt.hour.astype(str)
    sample["pickup_weekday"] = local.dt.dayofweek.astype(str)
    sample["weather_class"] = sample["weather_class"].fillna("unknown")
    assert len(sample) == sample_size and sample["row_id"].is_unique
    (root / "reports").mkdir(exist_ok=True)
    pd.DataFrame(monthly_counts).to_csv(root / "reports/source_counts.csv", index=False)
    sample.to_parquet(root / "outputs/assignment_sample.parquet", index=False)
    return sample


def split_data(data: pd.DataFrame, root: Path) -> dict[str, pd.DataFrame]:
    """Use a whole-hour chronological 80/20 boundary and exclude crossing training labels."""
    ordered = data.sort_values("pickup_utc")
    boundary = ordered["pickup_utc"].iloc[int(len(ordered) * 0.8)].ceil("h")
    earlier = ordered.loc[ordered["pickup_utc"].lt(boundary)]
    train = earlier.loc[earlier["dropoff_utc"].lt(boundary)].copy()
    validation = ordered.loc[ordered["pickup_utc"].ge(boundary)].copy()
    assert train["dropoff_utc"].max() < validation["pickup_utc"].min()
    assert set(train["weather_time_utc"]).isdisjoint(validation["weather_time_utc"])
    splits = {"train": train, "validation": validation}
    summary = {"validation_start_local": str(boundary.tz_convert("America/New_York")),
               "training_rows": len(train), "validation_rows": len(validation),
               "training_percent": 100 * len(train) / (len(train) + len(validation)),
               "boundary_crossing_training_rows": len(earlier) - len(train)}
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/split.json").write_text(json.dumps(summary, indent=2))
    month_counts = pd.DataFrame({name: part["pickup_utc"].dt.tz_convert("America/New_York").dt.month.value_counts()
                                 for name, part in splits.items()}).fillna(0).astype(int).sort_index()
    month_counts.index.name = "pickup_month"
    month_counts.to_csv(root / "reports/month_counts.csv")
    return splits


def prepare_inputs(splits: dict[str, pd.DataFrame], variant: str = "full", with_weather: bool = True) -> tuple[object, dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """Fit imputation, 0-1 scaling, and category encoding on training rows only."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    numeric = NUMERIC.copy()
    categorical = CATEGORICAL.copy()
    if variant == "distance_only":
        numeric = ["log_trip_distance"]
        categorical = []
        with_weather = False
    elif variant == "no_distance":
        numeric.remove("log_trip_distance")
    if variant == "no_weather":
        with_weather = False
    if with_weather:
        numeric.extend(WEATHER)
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=variant != "distance_only", keep_empty_features=True)),
        ("scaler", MinMaxScaler(feature_range=(0, 1), clip=True)),
    ])
    transformers = [("numeric", numeric_pipeline, numeric)]
    if categorical:
        categories = [[str(value) for value in range(264)], [str(value) for value in range(264)],
                      [str(value) for value in range(24)], [str(value) for value in range(7)]]
        encoder = OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        transformers.append(("categories", encoder, categorical))
    preprocessing = ColumnTransformer(transformers, sparse_threshold=0)
    preprocessing.fit(splits["train"])
    inputs = {name: preprocessing.transform(part).astype("float32") for name, part in splits.items()}
    targets = {name: part[TARGET].to_numpy(dtype="float32") for name, part in splits.items()}
    for matrix in inputs.values():
        assert np.isfinite(matrix).all() and matrix.min() >= 0 and matrix.max() <= 1.000001
    return preprocessing, inputs, targets, numeric + categorical


def duration_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate primary all-row and additional at-most-180-minute metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    metrics = {}
    for prefix, mask in [("all", np.ones(len(actual), dtype=bool)), ("upto180", actual <= 180)]:
        metrics[prefix + "_mae"] = float(mean_absolute_error(actual[mask], predicted[mask]))
        metrics[prefix + "_rmse"] = float(np.sqrt(mean_squared_error(actual[mask], predicted[mask])))
        metrics[prefix + "_r2"] = float(r2_score(actual[mask], predicted[mask]))
        metrics[prefix + "_rows"] = int(mask.sum())
    error = predicted - actual
    tail = actual > 180
    squared_total = np.square(error).sum()
    absolute_total = np.abs(error).sum()
    metrics["tail_squared_error_percent"] = float(100 * np.square(error[tail]).sum() / squared_total) if squared_total else 0.0
    metrics["tail_absolute_error_percent"] = float(100 * np.abs(error[tail]).sum() / absolute_total) if absolute_total else 0.0
    metrics["tail_mae_contribution_minutes"] = float(np.abs(error[tail]).sum() / len(actual))
    metrics["tail_rows"] = int(tail.sum())
    metrics["p90_absolute_error"] = float(np.quantile(np.abs(error), 0.9))
    metrics["negative_predictions"] = int((predicted < 0).sum())
    return metrics


def experiment_grid() -> list[dict]:
    """Return all 36 required model/loss/optimizer/learning-rate configurations."""
    import itertools
    return [dict(architecture=architecture, loss=loss, optimizer=optimizer, learning_rate=rate,
                 seed=SEED, feature_variant="full", with_weather=True, epochs=100, batch_size=32)
            for architecture, loss, optimizer, rate in itertools.product(
                ["linear", "mlp", "dnn"], ["mse", "mae"], ["sgd", "adam", "rmsprop"], [0.001, 0.01])]


def run_identifier(config: dict) -> str:
    """Create a readable unique ID including model settings, seed, and feature variant."""
    rate = format(config["learning_rate"], ".6g").replace(".", "p")
    dropout = config.get("dropout", 0)
    suffix = "_dropout" + str(dropout).replace(".", "p") if dropout else ""
    variant = config["feature_variant"]
    if variant == "full" and not config.get("with_weather", True):
        variant = "no_weather"
    return f"{config['architecture']}_{config['loss']}_{config['optimizer']}_lr{rate}_seed{config['seed']}_{variant}{suffix}"


def build_tensorflow(config: dict, width: int) -> object:
    """Build a linear model, 64-unit MLP, or 64/32-unit DNN in Keras."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(config["seed"])
    layers = [tf.keras.layers.Input(shape=(width,))]
    if config["architecture"] in ["mlp", "dnn"]:
        layers.append(tf.keras.layers.Dense(64, activation="relu"))
        if config.get("dropout", 0):
            layers.append(tf.keras.layers.Dropout(config["dropout"]))
    if config["architecture"] == "dnn":
        layers.append(tf.keras.layers.Dense(32, activation="relu"))
        if config.get("dropout", 0):
            layers.append(tf.keras.layers.Dropout(config["dropout"]))
    layers.append(tf.keras.layers.Dense(1))
    model = tf.keras.Sequential(layers)
    optimizer_types = {"sgd": tf.keras.optimizers.SGD, "adam": tf.keras.optimizers.Adam, "rmsprop": tf.keras.optimizers.RMSprop}
    optimizer = optimizer_types[config["optimizer"]](learning_rate=config["learning_rate"])
    model.compile(optimizer=optimizer, loss=config["loss"], metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")], steps_per_execution=64)
    return model


def fit_tensorflow(config: dict, inputs: dict[str, np.ndarray], targets: dict[str, np.ndarray], features: list[str], root: Path) -> dict:
    """Fit 100 epochs and cache only matching configurations, features, and input widths."""
    import tensorflow as tf
    import matplotlib.pyplot as plt
    config = {key: value.item() if isinstance(value, np.generic) else value for key, value in config.items()}
    assert config["epochs"] == 100 and config["batch_size"] == 32
    folder = root / "outputs/runs" / run_identifier(config)
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    width = inputs["train"].shape[1]
    context = {"training_rows": len(targets["train"]), "validation_rows": len(targets["validation"]), "period": "2020-01 through 2020-11", "sample_seed": SEED}
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        matches = cached.get("config") == config and cached.get("input_width") == width and cached.get("feature_list") == features and cached.get("data_context") == context
        if matches and cached.get("status") in ["complete", "failed"]:
            return cached
    result = {"run_id": folder.name, "config": config, **config, "input_width": width,
              "feature_list": features, "data_context": context, "status": "running"}
    result_path.write_text(json.dumps(result, indent=2))
    started = time.perf_counter()
    history = None
    try:
        tf.keras.backend.clear_session()
        model = build_tensorflow(config, width)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        training = tf.data.Dataset.from_tensor_slices((inputs["train"], targets["train"]))
        training = training.shuffle(len(targets["train"]), seed=config["seed"]).batch(32).with_options(options)
        validation = tf.data.Dataset.from_tensor_slices((inputs["validation"], targets["validation"])).batch(32).with_options(options)
        callbacks = [tf.keras.callbacks.ModelCheckpoint(folder / "best.weights.h5", monitor="val_mae", mode="min", save_best_only=True, save_weights_only=True),
                     tf.keras.callbacks.TerminateOnNaN()]
        history = model.fit(training, validation_data=validation, epochs=100, shuffle=False, callbacks=callbacks, verbose=0)
        curves = pd.DataFrame(history.history)
        curves.index = np.arange(1, len(curves) + 1)
        curves.index.name = "epoch"
        curves.to_csv(folder / "history.csv")
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(curves.index, curves["loss"], label="Training loss")
        axis.plot(curves.index, curves["val_loss"], label="Validation loss")
        axis.set(xlabel="Epoch", ylabel=config["loss"].upper(), title=f"{config['architecture'].upper()} | {config['optimizer']} | lr={config['learning_rate']}")
        axis.legend()
        figure.tight_layout()
        figure.savefig(folder / "loss.png", dpi=130)
        plt.close(figure)
        if len(curves) != 100 or not np.isfinite(curves[["loss", "val_loss"]]).all().all():
            raise ValueError("Numerical divergence; this run did not complete 100 finite epochs")
        model.load_weights(folder / "best.weights.h5")
        predictions = model.predict(validation, verbose=0).reshape(-1)
        np.save(folder / "validation_predictions.npy", predictions)
        model.save(folder / "best.keras")
        result.update(duration_metrics(targets["validation"], predictions))
        result.update(status="complete", epochs_completed=100, best_epoch=int(curves["val_mae"].idxmin()),
                      final_epoch_validation_mae=float(curves["val_mae"].iloc[-1]), final_epoch_training_loss=float(curves["loss"].iloc[-1]),
                      best_epoch_training_loss=float(curves.loc[curves["val_mae"].idxmin(), "loss"]))
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}", epochs_completed=0 if history is None else len(history.epoch))
    result["seconds"] = time.perf_counter() - started
    result_path.write_text(json.dumps(result, indent=2))
    return result




def error_analysis(frame: pd.DataFrame, predictions: np.ndarray, root: Path) -> dict[str, pd.DataFrame]:
    """Save large errors and distance/hour/rain subgroup MAE without dropping trips."""
    review = frame[["row_id", "pickup_utc", "dropoff_utc", "trip_distance", "pickup_hour", "rain", "snowfall", "precipitation", TARGET]].copy()
    review["prediction"] = predictions
    review["absolute_error"] = np.abs(review[TARGET] - predictions)
    review["distance_bucket"] = pd.cut(review["trip_distance"], [-np.inf, 0, 1, 3, 5, 10, 20, np.inf], labels=["<=0", "0-1", "1-3", "3-5", "5-10", "10-20", ">20"])
    rain_group = np.select([review[["rain", "snowfall", "precipitation"]].isna().any(axis=1), review["rain"].gt(0), review["precipitation"].eq(0)],
                           ["unknown", "rain", "dry"], default="snow_or_other")
    review["rain_group"] = rain_group
    review["distance_bucket"] = review["distance_bucket"].astype(object).fillna("unknown")
    outputs = {}
    for name, column in [("distance", "distance_bucket"), ("hour", "pickup_hour"), ("rain", "rain_group")]:
        outputs[name] = review.groupby(column, observed=False)["absolute_error"].agg(trips="size", mae_minutes="mean")
        if name == "hour":
            outputs[name] = outputs[name].sort_index(key=lambda values: values.astype(int))
        outputs[name].to_csv(root / "reports" / f"error_by_{name}.csv")
    outputs["largest_errors"] = review.nlargest(20, "absolute_error")
    outputs["largest_errors"].to_csv(root / "reports/largest_20_errors.csv", index=False)
    return outputs


def save_prediction_plots(actual: np.ndarray, predicted: np.ndarray, root: Path) -> None:
    """Save focused actual/predicted and all-row residual plots."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    limit = max(1.0, float(np.quantile(actual, 0.99)))
    axes[0].scatter(actual, predicted, s=7, alpha=0.15)
    axes[0].plot([0, limit], [0, limit], "k--")
    axes[0].set(xlim=(0, limit), ylim=(0, limit), xlabel="Actual minutes", ylabel="Predicted minutes", title="Focused axes: 99th percentile of actual duration")
    normal = actual <= 180
    axes[1].hist((predicted - actual)[normal], bins=80)
    axes[1].set(xlabel="Prediction minus actual (minutes)", ylabel="Trips", title="Residuals: validation trips up to 180 minutes")
    figure.tight_layout()
    figure.savefig(root / "reports/predictions.png", dpi=130)
    plt.close(figure)


def fit_pytorch(config: dict, inputs: dict[str, np.ndarray], targets: dict[str, np.ndarray], features: list[str], root: Path) -> dict:
    """Train the matching 64/32 PyTorch DNN for 100 epochs and retain its best checkpoint."""
    import torch
    import matplotlib.pyplot as plt
    config = {key: value.item() if isinstance(value, np.generic) else value for key, value in config.items()}
    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = root / "outputs/pytorch"
    folder.mkdir(exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        if cached.get("config") == config and cached.get("input_width") == inputs["train"].shape[1] and cached.get("feature_list") == features and cached.get("epochs_completed") == 100:
            return cached
    model = torch.nn.Sequential(torch.nn.Linear(inputs["train"].shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).to(device)
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], eps=1e-7)
    elif config["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"], alpha=0.9, eps=1e-7)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.L1Loss() if config["loss"] == "mae" else torch.nn.MSELoss()
    training_x = torch.from_numpy(inputs["train"])
    training_y = torch.from_numpy(targets["train"]).reshape(-1, 1)
    validation_x = torch.from_numpy(inputs["validation"]).to(device)
    validation_y = torch.from_numpy(targets["validation"]).reshape(-1, 1).to(device)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(training_x, training_y), batch_size=32, shuffle=True,
                                        generator=torch.Generator().manual_seed(config["seed"]))
    best_mae = float("inf")
    records = []
    started = time.perf_counter()
    for epoch in range(100):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise ValueError("PyTorch numerical divergence")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_x)
        model.eval()
        with torch.no_grad():
            predicted = model(validation_x)
            validation_loss = float(criterion(predicted, validation_y))
            validation_mae = float(torch.abs(predicted - validation_y).mean())
        records.append({"epoch": epoch + 1, "loss": total_loss / len(training_y), "val_loss": validation_loss, "val_mae": validation_mae})
        if validation_mae < best_mae:
            best_mae = validation_mae
            torch.save(model.state_dict(), folder / "best.pt")
    model.load_state_dict(torch.load(folder / "best.pt", weights_only=True, map_location=device))
    model.eval()
    with torch.no_grad():
        predictions = model(validation_x).cpu().numpy().reshape(-1)
    np.save(folder / "validation_predictions.npy", predictions)
    history = pd.DataFrame(records)
    history.to_csv(folder / "history.csv", index=False)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history["epoch"], history["loss"], label="Training loss")
    axis.plot(history["epoch"], history["val_loss"], label="Validation loss")
    axis.set(xlabel="Epoch", ylabel=config["loss"].upper(), title="PyTorch DNN")
    axis.legend()
    figure.tight_layout()
    figure.savefig(folder / "loss.png", dpi=130)
    plt.close(figure)
    result = {"framework": "PyTorch", "config": config, "feature_list": features, "input_width": inputs["train"].shape[1],
              "epochs_completed": 100, "best_epoch": int(history["val_mae"].idxmin()) + 1, "seconds": time.perf_counter() - started,
              **duration_metrics(targets["validation"], predictions)}
    result_path.write_text(json.dumps(result, indent=2))
    return result
