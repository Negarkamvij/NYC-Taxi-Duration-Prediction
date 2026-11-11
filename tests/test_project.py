"""Checks for data leakage, feature scope, scaling, tail metrics and required models."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd
import taxi_workflow as project


class ProjectTests(unittest.TestCase):
    """Protect the project's data and experiment requirements."""

    def test_removed_time_features(self) -> None:
        """Month and year-progress inputs must not enter any feature matrix."""
        self.assertNotIn("pickup_month", project.CATEGORICAL)
        self.assertNotIn("days_since_year_start", project.NUMERIC)

    def test_variant_ids_do_not_collide(self) -> None:
        """Weather, distance-free and seed variants receive separate readable IDs."""
        config = project.experiment_grid()[0]
        identifiers = {
            project.run_identifier(config),
            project.run_identifier(dict(config, with_weather=False)),
            project.run_identifier(dict(config, feature_variant="no_distance")),
            project.run_identifier(dict(config, seed=673)),
        }
        self.assertEqual(len(identifiers), 4)

    def test_tail_metrics_keep_primary_rows(self) -> None:
        """The 180-minute view must not replace all-row error calculations."""
        actual = np.array([10., 20., 1440.])
        predicted = np.array([10., 18., 20.])
        metrics = project.duration_metrics(actual, predicted)
        self.assertEqual(metrics["all_rows"], 3)
        self.assertEqual(metrics["upto180_rows"], 2)
        self.assertEqual(metrics["tail_rows"], 1)
        self.assertAlmostEqual(metrics["all_mae"], 474.)
        self.assertGreater(metrics["tail_squared_error_percent"], 99.)

    def test_scaling_fits_training_only(self) -> None:
        """Validation values cannot alter imputation or fitted numeric ranges."""
        fields = {name: ["1", "2"] for name in project.CATEGORICAL}
        fields.update({name: [1., 3.] for name in project.NUMERIC + project.WEATHER})
        fields[project.TARGET] = [10., 20.]
        train = pd.DataFrame(fields)
        validation = train.copy()
        validation["passenger_count"] = 1000.
        transformer, inputs, targets, features = project.prepare_inputs({"train": train, "validation": validation})
        scaler = transformer.named_transformers_["numeric"].named_steps["scaler"]
        self.assertEqual(scaler.data_max_[0], 3.)
        self.assertLessEqual(inputs["validation"].max(), 1.)
        self.assertNotIn(project.TARGET, features)
        _, distance_inputs, _, distance_features = project.prepare_inputs({"train": train, "validation": validation}, variant="distance_only")
        self.assertEqual(distance_inputs["train"].shape[1], 1)
        self.assertEqual(distance_features, ["log_trip_distance"])

    def test_chronological_split(self) -> None:
        """Training trips finish before validation and weather hours are not shared."""
        pickup = pd.date_range("2020-01-01", periods=25, freq="12h", tz="UTC")
        frame = pd.DataFrame({"pickup_utc": pickup, "dropoff_utc": pickup + pd.Timedelta(minutes=10)})
        frame["weather_time_utc"] = frame["pickup_utc"].dt.floor("h")
        frame.loc[3, "dropoff_utc"] = pd.Timestamp("2020-02-01", tz="UTC")
        with TemporaryDirectory() as folder:
            splits = project.split_data(frame, Path(folder))
        self.assertNotIn(3, splits["train"].index)
        self.assertLess(splits["train"]["dropoff_utc"].max(), splits["validation"]["pickup_utc"].min())
        self.assertTrue(set(splits["train"]["weather_time_utc"]).isdisjoint(splits["validation"]["weather_time_utc"]))
        self.assertEqual(len(frame), 25)

    def test_required_grid(self) -> None:
        """The complete 36-run grid retains 100 epochs and batch size 32."""
        configurations = project.experiment_grid()
        self.assertEqual(len(configurations), 36)
        self.assertTrue(all(config["epochs"] == 100 and config["batch_size"] == 32 for config in configurations))
        for architecture in ["linear", "mlp", "dnn"]:
            subset = [config for config in configurations if config["architecture"] == architecture]
            self.assertEqual(len(subset), 12)
            self.assertEqual({config["loss"] for config in subset}, {"mae", "mse"})
            self.assertEqual({config["optimizer"] for config in subset}, {"sgd", "adam", "rmsprop"})

    def test_dst_times_are_not_guessed(self) -> None:
        """Ambiguous and nonexistent local times are explicitly unavailable."""
        result = project.local_utc(pd.Series(["2020-01-01 10:00", "2020-03-08 02:30", "2020-11-01 01:30"]))
        self.assertEqual(str(result.iloc[0]), "2020-01-01 15:00:00+00:00")
        self.assertTrue(result.iloc[1:].isna().all())


if __name__ == "__main__":
    unittest.main()
