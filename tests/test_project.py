"""Validation of the project's data split, scaling, architectures, and experiment grid."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import taxi_workflow as project


class ProjectTests(unittest.TestCase):
    """Check requirements which affect model validity."""

    def test_missing_weather_remains_unknown(self) -> None:
        """Do not convert missing precipitation into dry weather."""
        data=pd.DataFrame({'rain':[np.nan,0.,1.,0.,1.], 'snowfall':[0.,0.,0.,1.,1.], 'precipitation':[np.nan,0.,1.,1.,2.]})
        self.assertEqual(project.weather_classes(data).tolist(),['unknown','dry','rain','snow','mixed'])

    def test_daylight_saving_timestamps(self) -> None:
        """Flag local times which cannot be converted unambiguously."""
        utc=project.local_utc(pd.Series(['2020-01-01 10:00','2020-03-08 02:30','2020-11-01 01:30']))
        self.assertEqual(str(utc.iloc[0]),'2020-01-01 15:00:00+00:00')
        self.assertTrue(utc.iloc[1:].isna().all())

    def test_chronological_split(self) -> None:
        """Training labels finish before validation begins, without shared weather hours."""
        pickup=pd.date_range('2020-01-01',periods=25,freq='12h',tz='UTC')
        data=pd.DataFrame({'pickup_utc':pickup,'dropoff_utc':pickup+pd.Timedelta(minutes=10)})
        data['weather_time_utc']=data.pickup_utc.dt.floor('h')
        data.loc[3,'dropoff_utc']=pd.Timestamp('2020-02-01',tz='UTC')
        with TemporaryDirectory() as folder:
            result=project.split_assignment(data,Path(folder))
        self.assertNotIn(3,result['train'].index)
        self.assertLess(result['train'].dropoff_utc.max(),result['validation'].pickup_utc.min())
        self.assertTrue(set(result['train'].weather_time_utc).isdisjoint(result['validation'].weather_time_utc))
        self.assertEqual(len(data),25)

    def test_training_only_minmax_scaling(self) -> None:
        """Validation extremes must not alter training ranges or exceed the clipped 0-1 inputs."""
        base={name:['1','2'] for name in project.CATEGORICAL}
        base.update({name:[1.,3.] for name in project.NUMERIC+project.WEATHER})
        base[project.TARGET]=[10.,20.]
        train=pd.DataFrame(base);validation=train.copy();validation['passenger_count']=1000.
        transformer,arrays,targets=project.prepare_inputs({'train':train,'validation':validation})
        scaler=transformer.named_transformers_['numeric'].named_steps['scale']
        self.assertEqual(scaler.data_min_[0],1.)
        self.assertEqual(scaler.data_max_[0],3.)
        self.assertGreaterEqual(arrays['validation'].min(),0.)
        self.assertLessEqual(arrays['validation'].max(),1.)
        self.assertFalse(any(project.TARGET in name or 'fare' in name for name in transformer.get_feature_names_out()))

    def test_required_experiment_settings(self) -> None:
        """All three models cover both losses, three optimizers, and two learning rates."""
        grid=project.experiment_grid()
        self.assertEqual(len(grid),36)
        for architecture in ['linear','mlp','dnn']:
            group=[c for c in grid if c['architecture']==architecture]
            self.assertEqual(len(group),12)
            self.assertEqual({c['loss'] for c in group},{'mse','mae'})
            self.assertEqual({c['optimizer'] for c in group},{'sgd','adam','rmsprop'})
            self.assertEqual({c['learning_rate'] for c in group},{.001,.01})
        self.assertTrue(all(c['epochs']==100 and c['batch_size']==32 for c in grid))

    def test_network_architectures(self) -> None:
        """Check the exact hidden-layer counts required by the assignment."""
        import tensorflow as tf
        for name,units in [('linear',[1]),('mlp',[64,1]),('dnn',[64,32,1])]:
            config=dict(architecture=name,loss='mae',optimizer='adam',learning_rate=.001,seed=672)
            model=project.build_tensorflow(config,10)
            self.assertEqual([layer.units for layer in model.layers if isinstance(layer,tf.keras.layers.Dense)],units)
            self.assertEqual(model.loss,'mae')


if __name__=='__main__':
    unittest.main()
