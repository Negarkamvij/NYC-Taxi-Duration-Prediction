"""Checks for leakage boundaries, weather missingness, feature scope, and grid coverage."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
import numpy as np
import taxi_workflow as project


class WorkflowTests(unittest.TestCase):
    """Protect decisions which materially affect the validity of the experiment."""

    def test_missing_precipitation_is_not_dry(self) -> None:
        """Missing weather stays unknown rather than becoming a fabricated dry hour."""
        weather=pd.DataFrame({'rain':[np.nan,0.,1.,0.,1.], 'snowfall':[0.,0.,0.,1.,1.], 'precipitation':[np.nan,0.,1.,1.,2.]})
        self.assertEqual(project.weather_classes(weather).tolist(),['unknown','dry','rain','snow','mixed'])

    def test_dst_labels_are_not_guessed(self) -> None:
        """Ambiguous/nonexistent local timestamps are marked unavailable explicitly."""
        source=pd.Series(['2020-01-01 10:00','2020-03-08 02:30','2020-11-01 01:30'])
        utc=project.local_utc(source)
        self.assertEqual(str(utc.iloc[0]),'2020-01-01 15:00:00+00:00')
        self.assertTrue(utc.iloc[1:].isna().all())

    def test_complete_weather_hours_and_rides_do_not_cross_split(self) -> None:
        """Train labels and weather groups stop before validation, which stops before test."""
        pickup=pd.DatetimeIndex(list(pd.date_range('2020-01-01',periods=20,freq='12h',tz='UTC'))+list(pd.date_range('2020-12-01 12:00',periods=3,freq='h',tz='UTC')))
        frame=pd.DataFrame({'pickup_utc':pickup,'dropoff_utc':pickup+pd.Timedelta(minutes=10)})
        frame['weather_time_utc']=frame.pickup_utc.dt.floor('h');frame['weather_class']='dry'
        # Force an earlier-pickup trip to finish after validation starts.
        frame.loc[3,'dropoff_utc']=pd.Timestamp('2020-11-30',tz='UTC')
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/'reports').mkdir()
            split=project.split_by_time(frame,root)
        self.assertNotIn(3,split['train'].index)
        self.assertLess(split['train'].dropoff_utc.max(),split['validation'].pickup_utc.min())
        self.assertTrue(set(split['train'].weather_time_utc).isdisjoint(split['validation'].weather_time_utc))

    def test_preprocessor_fits_training_only(self) -> None:
        """Validation extremes do not change scaler parameters; post-trip fields are absent."""
        base={name:['1','2'] for name in project.CATEGORICAL}
        base.update({name:[1.,3.] for name in project.NUMERIC+project.WEATHER})
        base[project.TARGET]=[10.,20.]
        train=pd.DataFrame(base);validation=train.copy()
        validation['passenger_count']=1000.
        transformer,arrays,targets=project.prepare_inputs({'train':train,'validation':validation,'test':validation})
        scaler=transformer.named_transformers_['numeric'].named_steps['scale']
        self.assertEqual(scaler.mean_[0],2.)
        names=transformer.get_feature_names_out()
        self.assertFalse(any(project.TARGET in name or 'trip_distance' in name or 'fare' in name for name in names))
        self.assertTrue(np.isfinite(arrays['validation']).all())

    def test_grid_covers_required_settings(self) -> None:
        """Every architecture has both losses, all optimizers, and two learning rates."""
        grid=project.experiment_grid()
        self.assertEqual(len(grid),36)
        self.assertTrue(all(c['epochs']==100 and c['batch_size']==32 for c in grid))
        for architecture in ['linear','mlp','dnn']:
            group=[c for c in grid if c['architecture']==architecture]
            self.assertEqual(len(group),12)
            self.assertEqual({c['loss'] for c in group},{'mse','mae'})
            self.assertEqual({c['optimizer'] for c in group},{'sgd','adam','rmsprop'})


if __name__=='__main__':
    unittest.main()
