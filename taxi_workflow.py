"""CS672 taxi-duration workflow: preserve sources and evaluate time-ordered models."""
from __future__ import annotations
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from datetime import datetime, timezone
import hashlib, json, shutil, time
import numpy as np
import pandas as pd

SEED = 672
WEATHER = ['temperature_2m','relative_humidity_2m','precipitation','rain','snowfall','wind_speed_10m','surface_pressure']
NUMERIC = ['passenger_count','days_since_year_start']
CATEGORICAL = ['pickup_zone','destination_zone','pickup_hour','pickup_weekday','pickup_month']
TARGET = 'duration_minutes'


def file_hash(path: Path) -> str:
    """Return a file's SHA-256 digest without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, path: Path) -> Path:
    """Download a missing file atomically; preserve existing local source files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = Path(str(path)+'.part')
        for attempt in range(3):
            try:
                with urlopen(Request(url, headers={'User-Agent':'CS672-taxi-study'}), timeout=120) as response, partial.open('wb') as output:
                    shutil.copyfileobj(response, output)
                partial.replace(path)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2)
    return path


def load_weather(root: Path) -> pd.DataFrame:
    """Download/cache full-year ERA5 and return hourly UTC values with provenance.

    Precipitation describes the preceding hour. Reanalysis is retrospective and
    does not establish the real-time availability of these weather measurements.
    """
    path = root/'data/weather_era5_2020.json'
    params = dict(latitude=40.7075, longitude=-74.0113, start_date='2019-12-31', end_date='2021-01-02',
                  hourly=','.join(WEATHER), models='era5', timezone='UTC', temperature_unit='celsius',
                  wind_speed_unit='kmh', precipitation_unit='mm')
    url = 'https://archive-api.open-meteo.com/v1/archive?'+urlencode(params)
    download_file(url,path)
    payload=json.loads(path.read_text())
    units={'temperature_2m':'°C','relative_humidity_2m':'%','precipitation':'mm','rain':'mm','snowfall':'cm','wind_speed_10m':'km/h','surface_pressure':'hPa'}
    assert all(payload['hourly_units'][k]==v for k,v in units.items())
    weather=pd.DataFrame(payload['hourly'])
    weather['weather_time_utc']=pd.to_datetime(weather.pop('time'),utc=True)
    weather=weather.sort_values('weather_time_utc').reset_index(drop=True)
    assert weather.weather_time_utc.is_unique
    assert weather.weather_time_utc.diff().dropna().eq(pd.Timedelta(hours=1)).all()
    weather['weather_class']=weather_classes(weather)
    provenance=dict(url=url,dataset='ERA5 reanalysis via Open-Meteo',requested_location=[40.7075,-74.0113],
                    returned_grid_location=[payload['latitude'],payload['longitude']],units=units,
                    downloaded_at_or_first_recorded_utc=datetime.now(timezone.utc).isoformat(),
                    limitation='Retrospective reconstruction, not an as-issued forecast or live-feed backtest',sha256=file_hash(path))
    report=root/'reports';report.mkdir(exist_ok=True)
    if not (report/'weather_provenance_2020.json').exists():
        (report/'weather_provenance_2020.json').write_text(json.dumps(provenance,indent=2))
    local=weather.weather_time_utc.dt.tz_convert('America/New_York')
    coverage=weather.loc[local.dt.year.eq(2020)].copy()
    coverage['month']=local.loc[coverage.index].dt.month
    coverage.groupby(['month','weather_class']).size().rename('distinct_hours').reset_index().to_csv(report/'weather_coverage_2020.csv',index=False)
    return weather


def weather_classes(frame: pd.DataFrame) -> pd.Series:
    """Return exclusive weather labels, keeping missing precipitation unknown."""
    missing=frame[['rain','snowfall','precipitation']].isna().any(axis=1)
    rain=frame.rain.gt(0);snow=frame.snowfall.gt(0)
    return pd.Series(np.select([missing,rain&snow,snow,rain,frame.precipitation.eq(0)],
                              ['unknown','mixed','snow','rain','dry'],default='other_or_trace'),index=frame.index)


def local_utc(values: pd.Series) -> pd.Series:
    """Convert naive New York timestamps to UTC; flag unresolved DST times as NaT."""
    return pd.to_datetime(values,errors='coerce').dt.tz_localize('America/New_York',ambiguous='NaT',nonexistent='NaT').dt.tz_convert('UTC')


def prepare_sample(root: Path, weather: pd.DataFrame, sample_size: int = 50000, test_size: int = 10000) -> pd.DataFrame:
    """Audit every monthly source row and uniformly sample eligible trips.

    Jan-Nov form one uniform-priority sample; December forms a separate uniform
    test sample. Original files stay intact. Unknown zones remain categories.
    Eligibility requires valid UTC times, positive duration and intended month;
    long rides are flagged but not discarded by a duration or distance threshold.
    """
    report=root/'reports';report.mkdir(exist_ok=True)
    pool=pd.DataFrame();test=pd.DataFrame();audits=[];manifest=[]
    cols=['tpep_pickup_datetime','tpep_dropoff_datetime','PULocationID','DOLocationID','passenger_count','trip_distance']
    for month in range(1,13):
        name=f'yellow_tripdata_2020-{month:02d}.parquet'
        path=download_file('https://d37ci6vzurychx.cloudfront.net/trip-data/'+name,root/'data'/name)
        raw=pd.read_parquet(path,columns=cols)
        pickup=local_utc(raw.tpep_pickup_datetime);dropoff=local_utc(raw.tpep_dropoff_datetime)
        duration=(dropoff-pickup).dt.total_seconds()/60
        local=pickup.dt.tz_convert('America/New_York')
        in_month=local.dt.year.eq(2020)&local.dt.month.eq(month)
        valid=pickup.notna()&dropoff.notna()&np.isfinite(duration)&duration.gt(0)&in_month
        audit={'month':month,'source_rows':len(raw),'eligible_rows':int(valid.sum()),'ineligible_not_deleted':int((~valid).sum()),
               'missing_or_unresolved_time':int((pickup.isna()|dropoff.isna()).sum()),'outside_intended_month':int((~in_month).sum()),
               'nonpositive_or_unavailable_duration':int((~np.isfinite(duration)|duration.le(0)).sum()),
               'long_duration_flag':int(duration.gt(180).sum()),'negative_or_nonfinite_distance':int((raw.trip_distance.lt(0)|~np.isfinite(raw.trip_distance)).sum()),
               'long_distance_flag':int(raw.trip_distance.gt(100).sum()),'source_rows_removed':0}
        audits.append(audit)
        # Random priorities do not depend on trip duration or weather.
        priority=np.random.default_rng(SEED+month).random(len(raw))
        indices=np.flatnonzero(valid.to_numpy());cap=test_size if month==12 else sample_size
        if len(indices)>cap:
            indices=indices[np.argpartition(priority[indices],cap-1)[:cap]]
        candidate=pd.DataFrame({'row_id':[f'{month:02d}:{i}' for i in indices],
                                'pickup_utc':pickup.iloc[indices].to_numpy(),'dropoff_utc':dropoff.iloc[indices].to_numpy(),
                                TARGET:duration.iloc[indices].to_numpy(),'priority':priority[indices],
                                'passenger_count':raw.passenger_count.iloc[indices].to_numpy()})
        for source,dest in [('PULocationID','pickup_zone'),('DOLocationID','destination_zone')]:
            zones=raw[source].iloc[indices]
            candidate[dest]=zones.where(zones.between(1,263)&zones.mod(1).eq(0),0).fillna(0).astype(int).astype(str).to_numpy()
        candidate['passenger_count']=candidate.passenger_count.where(candidate.passenger_count.ge(0))
        if month==12:
            test=candidate
        else:
            pool=pd.concat([pool,candidate],ignore_index=True).nsmallest(sample_size,'priority')
        manifest.append({'month':month,'file':name,'sha256':file_hash(path),'source_rows':len(raw),'eligible_rows':int(valid.sum())})
        print(f'Month {month:02d}: {len(raw):,} source rows, {int(valid.sum()):,} eligible; source rows removed: 0',flush=True)
        del raw,pickup,dropoff,duration,priority,local
    frame=pd.concat([pool,test],ignore_index=True).sort_values('pickup_utc').reset_index(drop=True)
    frame['pickup_utc']=pd.to_datetime(frame.pickup_utc,utc=True)
    frame['dropoff_utc']=pd.to_datetime(frame.dropoff_utc,utc=True)
    assert frame.row_id.is_unique
    frame['weather_time_utc']=frame.pickup_utc.dt.floor('h')
    frame=frame.merge(weather,on='weather_time_utc',how='left',validate='many_to_one')
    assert len(frame)==len(pool)+len(test)
    # A join timestamp is a measurement time, not a proven publication time.
    assert frame.weather_time_utc.le(frame.pickup_utc).all()
    frame['weather_class']=frame.weather_class.fillna('unmatched')
    local=frame.pickup_utc.dt.tz_convert('America/New_York')
    frame['pickup_hour']=local.dt.hour.astype(str)
    frame['pickup_weekday']=local.dt.dayofweek.astype(str)
    frame['pickup_month']=local.dt.month.astype(str)
    frame['days_since_year_start']=(local.dt.tz_localize(None)-pd.Timestamp('2020-01-01')).dt.total_seconds()/86400
    pd.DataFrame(audits).to_csv(report/'source_audit_2020.csv',index=False)
    (report/'data_manifest_2020.json').write_text(json.dumps({'seed':SEED,'sample_size_jan_nov':len(pool),'sample_size_december':len(test),'sampling':'Uniform random priorities over eligible source rows, not equal monthly quotas','files':manifest},indent=2))
    (root/'outputs').mkdir(exist_ok=True)
    frame.to_parquet(root/'outputs/model_sample.parquet',index=False)
    return frame


def split_by_time(frame: pd.DataFrame, root: Path) -> dict[str,pd.DataFrame]:
    """Make an approximately 80/20 Jan-Nov split and hold December for final test.

    Whole-hour boundaries prevent the same weather hour appearing in both sets.
    Purge earlier rides that finish across the next period's boundary. Source
    rows remain untouched; the exact split and exclusions are reported.
    """
    final_start=pd.Timestamp('2020-12-01',tz='America/New_York').tz_convert('UTC')
    development=frame.loc[frame.pickup_utc.lt(final_start)].sort_values('pickup_utc')
    boundary=development.pickup_utc.iloc[int(len(development)*.8)].ceil('h')
    train_before=development.loc[development.pickup_utc.lt(boundary)]
    validation_before=development.loc[development.pickup_utc.ge(boundary)]
    splits={'train':train_before.loc[train_before.dropoff_utc.lt(boundary)].copy(),
            'validation':validation_before.loc[validation_before.dropoff_utc.lt(final_start)].copy(),
            'test':frame.loc[frame.pickup_utc.ge(final_start)].copy()}
    assert splits['train'].dropoff_utc.max()<splits['validation'].pickup_utc.min()
    assert splits['validation'].dropoff_utc.max()<splits['test'].pickup_utc.min()
    assert set(splits['train'].weather_time_utc).isdisjoint(splits['validation'].weather_time_utc)
    assert set(splits['validation'].weather_time_utc).isdisjoint(splits['test'].weather_time_utc)
    summary={'validation_start_local':str(boundary.tz_convert('America/New_York')),'test_start_local':str(final_start.tz_convert('America/New_York')),
             'train':len(splits['train']),'validation':len(splits['validation']),'test':len(splits['test']),
             'train_boundary_purged':len(train_before)-len(splits['train']),
             'validation_boundary_purged':len(validation_before)-len(splits['validation']),
             'training_share_after_purge':len(splits['train'])/(len(splits['train'])+len(splits['validation']))}
    report=root/'reports';(report/'split_2020.json').write_text(json.dumps(summary,indent=2))
    exposure=[]
    for name,part in splits.items():
        for category,group in part.groupby('weather_class'):
            exposure.append({'split':name,'weather_class':category,'taxi_rows':len(group),
                             'distinct_hours':group.weather_time_utc.nunique(),
                             'distinct_days':group.pickup_utc.dt.tz_convert('America/New_York').dt.date.nunique()})
    pd.DataFrame(exposure).to_csv(report/'split_weather_2020.csv',index=False)
    return splits


def prepare_inputs(splits: dict[str,pd.DataFrame], with_weather: bool = True) -> tuple[object,dict[str,np.ndarray],dict[str,np.ndarray]]:
    """Fit numeric imputation/scaling and category encoding on training rows only."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler,OneHotEncoder
    numeric=NUMERIC+(WEATHER if with_weather else [])
    # Fixed calendar/zone domains are public definitions, not validation-derived vocabularies.
    categories=[[str(i) for i in range(264)],[str(i) for i in range(264)],
                [str(i) for i in range(24)],[str(i) for i in range(7)],[str(i) for i in range(1,13)]]
    preprocessing=ColumnTransformer([
        ('numeric',Pipeline([('missing',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]),numeric),
        ('category',OneHotEncoder(categories=categories,handle_unknown='ignore',sparse_output=False,dtype=np.float32),CATEGORICAL),
    ],sparse_threshold=0)
    preprocessing.fit(splits['train'])
    inputs={name:preprocessing.transform(part).astype('float32') for name,part in splits.items()}
    targets={name:part[TARGET].to_numpy(dtype='float32') for name,part in splits.items()}
    assert all(np.isfinite(x).all() for x in inputs.values())
    return preprocessing,inputs,targets


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str,float]:
    """Return common duration metrics in minutes without silently clipping predictions."""
    from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
    actual=np.asarray(actual).reshape(-1);predicted=np.asarray(predicted).reshape(-1)
    return {'mae_minutes':float(mean_absolute_error(actual,predicted)),
            'rmse_minutes':float(np.sqrt(mean_squared_error(actual,predicted))),
            'r2':float(r2_score(actual,predicted)),
            'p90_absolute_error_minutes':float(np.quantile(np.abs(actual-predicted),.9)),
            'negative_prediction_count':int((predicted<0).sum())}


def baseline_results(splits: dict[str,pd.DataFrame], root: Path) -> pd.DataFrame:
    """Evaluate two training-only baselines on validation, leaving test labels unused."""
    train=splits['train'];validation=splits['validation'];median=float(train[TARGET].median())
    routes=train.groupby(['pickup_zone','destination_zone'])[TARGET].agg(['median','count'])
    trusted=routes.loc[routes['count'].ge(20),'median']
    keys=pd.MultiIndex.from_frame(validation[['pickup_zone','destination_zone']])
    route_prediction=pd.Series(trusted.reindex(keys).to_numpy()).fillna(median).to_numpy()
    results=[]
    for name,pred in [('training_median',np.full(len(validation),median)),('training_route_median_min20',route_prediction)]:
        results.append({'model':name,'split':'validation',**metrics(validation[TARGET].to_numpy(),pred)})
    result=pd.DataFrame(results);result.to_csv(root/'reports/validation_baselines.csv',index=False)
    return result


def experiment_grid() -> list[dict]:
    """Return the explicit 36-run grid covering three architectures and required settings."""
    import itertools
    return [dict(architecture=a,loss=l,optimizer=o,learning_rate=lr,with_weather=True,epochs=100,batch_size=32,seed=SEED)
            for a,l,o,lr in itertools.product(['linear','mlp','dnn'],['mse','mae'],['sgd','adam','rmsprop'],[.001,.01])]


def build_tensorflow(config: dict, width: int) -> object:
    """Build a Keras linear, one-hidden-layer MLP, or two-hidden-layer DNN."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(config['seed'])
    layers=[tf.keras.layers.Input(shape=(width,))]
    if config['architecture'] in ['mlp','dnn']:
        layers.append(tf.keras.layers.Dense(64,activation='relu'))
    if config['architecture']=='dnn':
        layers.append(tf.keras.layers.Dense(32,activation='relu'))
    layers.append(tf.keras.layers.Dense(1))
    model=tf.keras.Sequential(layers)
    optimizer={'sgd':tf.keras.optimizers.SGD,'adam':tf.keras.optimizers.Adam,'rmsprop':tf.keras.optimizers.RMSprop}[config['optimizer']](learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer,loss=config['loss'],metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])
    return model


def fit_tensorflow(config: dict, inputs: dict[str,np.ndarray], targets: dict[str,np.ndarray], root: Path, fingerprint: str) -> dict:
    """Train a 100-epoch run, checkpoint by validation MAE and save all loss curves.

    A numerically failed run is recorded as failed, never as a completed run.
    The test arrays are not accessed. Existing successful runs can be resumed
    only when the data fingerprint and complete configuration agree.
    """
    import tensorflow as tf
    import matplotlib.pyplot as plt
    assert config['epochs']==100 and config['batch_size']==32
    import inspect
    definition_hash=hashlib.sha256((inspect.getsource(build_tensorflow)+inspect.getsource(prepare_inputs)).encode()).hexdigest()
    identity=hashlib.sha256(json.dumps({'config':config,'data':fingerprint},sort_keys=True).encode()).hexdigest()[:12]
    run_id=f"{config['architecture']}_{config['loss']}_{config['optimizer']}_{config['learning_rate']}_{identity}"
    folder=root/'outputs/runs'/run_id;folder.mkdir(parents=True,exist_ok=True)
    result_path=folder/'result.json'
    if result_path.exists():
        existing=json.loads(result_path.read_text())
        if existing.get('status')=='complete' and existing.get('definition_hash')==definition_hash:return existing
    config={k:(v.item() if isinstance(v,np.generic) else v) for k,v in config.items()}
    result={'run_id':run_id,**config,'hardware':('GPU' if tf.config.list_physical_devices('GPU') else 'CPU'),'data_fingerprint':fingerprint,'definition_hash':definition_hash,'status':'running'}
    result_path.write_text(json.dumps(result,indent=2))
    start=time.perf_counter();history=None
    try:
        tf.keras.backend.clear_session()
        model=build_tensorflow(config,inputs['train'].shape[1])
        options=tf.data.Options();options.threading.private_threadpool_size=1
        training=tf.data.Dataset.from_tensor_slices((inputs['train'],targets['train'])).shuffle(len(targets['train']),seed=config['seed']).batch(32).with_options(options)
        validation=tf.data.Dataset.from_tensor_slices((inputs['validation'],targets['validation'])).batch(32).with_options(options)
        checkpoint=folder/'best.weights.h5'
        history=model.fit(training,validation_data=validation,epochs=100,verbose=0,shuffle=False,callbacks=[
            tf.keras.callbacks.ModelCheckpoint(checkpoint,monitor='val_mae',mode='min',save_best_only=True,save_weights_only=True),
            tf.keras.callbacks.TerminateOnNaN(),
        ])
        curves=pd.DataFrame(history.history);curves.index=np.arange(1,len(curves)+1);curves.index.name='epoch'
        curves.to_csv(folder/'history.csv')
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot(curves.index,curves.loss,label='Training loss');ax.plot(curves.index,curves.val_loss,label='Validation loss')
        ax.set(xlabel='Epoch',ylabel=f"{config['loss'].upper()} ({'minutes squared' if config['loss']=='mse' else 'minutes'})",title=run_id)
        ax.legend();fig.tight_layout();fig.savefig(folder/'loss.png',dpi=140);plt.close(fig)
        if len(curves)!=100 or not np.isfinite(curves[['loss','val_loss']].to_numpy()).all():
            raise ValueError('Numerical failure: this run did not complete 100 finite epochs')
        model.load_weights(checkpoint)
        prediction=model.predict(validation,verbose=0).reshape(-1)
        result.update(status='complete',epochs_completed=100,best_epoch=int(curves.val_mae.idxmin()),final_epoch_validation_mae_minutes=float(curves.val_mae.iloc[-1]),
                      **{'validation_'+k:v for k,v in metrics(targets['validation'],prediction).items()})
        model.save(folder/'best.keras')
        np.save(folder/'validation_predictions.npy',prediction)
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}',epochs_completed=0 if history is None else len(history.epoch))
    result['seconds']=time.perf_counter()-start
    result_path.write_text(json.dumps(result,indent=2))
    return result


def collect_results(root: Path) -> pd.DataFrame:
    """Collect genuine logged run results; absent experiments are never fabricated."""
    result=pd.DataFrame([json.loads(p.read_text()) for p in (root/'outputs/runs').glob('*/result.json')])
    if not result.empty:
        if 'validation_mae_minutes' in result:result=result.sort_values('validation_mae_minutes')
        result.to_csv(root/'reports/experiment_results.csv',index=False)
    return result


def evaluate_selected(root: Path, result: dict, inputs: dict[str,np.ndarray], targets: dict[str,np.ndarray], splits: dict[str,pd.DataFrame]) -> pd.DataFrame:
    """Evaluate a frozen selected model on December once; cache and flag any prior test access.

    Call only after all required validation experiments and weather ablation.
    Never use this report to tune parameters and then call it an untouched test.
    """
    import tensorflow as tf
    marker=root/'reports/final_test_metrics.json'
    if marker.exists():raise RuntimeError('December was already evaluated. Keep the original report; further tuning is exploratory.')
    model=tf.keras.models.load_model(root/'outputs/runs'/result['run_id']/'best.keras')
    prediction=model.predict(inputs['test'],batch_size=32,verbose=0).reshape(-1)
    save_error_plots(targets['test'],prediction,root/'outputs','final_test')
    report={'selected_run_id':result['run_id'],'test_period':'December 2020',**metrics(targets['test'],prediction)}
    marker.write_text(json.dumps(report,indent=2))
    frame=splits['test'][['row_id','pickup_utc','weather_time_utc','weather_class',TARGET]].copy()
    frame['predicted_minutes']=prediction;frame['absolute_error']=np.abs(frame[TARGET]-prediction)
    frame.to_parquet(root/'outputs/final_test_predictions.parquet',index=False)
    grouped=frame.groupby('weather_class').agg(trips=(TARGET,'size'),weather_hours=('weather_time_utc','nunique'),mae_minutes=('absolute_error','mean'))
    grouped.to_csv(root/'reports/final_test_weather_errors.csv')
    return grouped


def fit_pytorch(config: dict, inputs: dict[str,np.ndarray], targets: dict[str,np.ndarray], root: Path) -> dict:
    """Train the extra-credit 64/32 DNN for 100 epochs with the same input arrays.

    Save the validation-MAE checkpoint and training/validation loss. Test arrays
    are not accessed. Matching seeds do not imply identical framework results.
    """
    import torch
    import matplotlib.pyplot as plt
    config={k:(v.item() if isinstance(v,np.generic) else v) for k,v in config.items()}
    assert config['architecture']=='dnn' and config['epochs']==100 and config['batch_size']==32
    torch.manual_seed(config['seed']);torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=torch.nn.Sequential(torch.nn.Linear(inputs['train'].shape[1],64),torch.nn.ReLU(),torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,1)).to(device)
    optimizer={'sgd':torch.optim.SGD,'adam':torch.optim.Adam,'rmsprop':torch.optim.RMSprop}[config['optimizer']](model.parameters(),lr=config['learning_rate'])
    # RMSprop defaults differ: match Keras rho=.9 and epsilon=1e-7; Adam epsilon too.
    if config['optimizer']=='rmsprop':optimizer=torch.optim.RMSprop(model.parameters(),lr=config['learning_rate'],alpha=.9,eps=1e-7)
    elif config['optimizer']=='adam':optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],eps=1e-7)
    loss_fn=torch.nn.MSELoss() if config['loss']=='mse' else torch.nn.L1Loss()
    x=torch.from_numpy(inputs['train']);y=torch.from_numpy(targets['train']).reshape(-1,1)
    xv=torch.from_numpy(inputs['validation']).to(device);yv=torch.from_numpy(targets['validation']).reshape(-1,1).to(device)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(config['seed']))
    folder=root/'outputs/pytorch';folder.mkdir(exist_ok=True)
    best=float('inf');history=[];start=time.perf_counter()
    for epoch in range(100):
        model.train();total=0.
        for xb,yb in loader:
            xb=xb.to(device);yb=yb.to(device)
            optimizer.zero_grad();prediction=model(xb);loss=loss_fn(prediction,yb)
            if not torch.isfinite(loss):raise ValueError('PyTorch numerical failure; run incomplete')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(xb)
        model.eval()
        with torch.no_grad():
            prediction=model(xv);val_loss=float(loss_fn(prediction,yv));val_mae=float(torch.mean(torch.abs(prediction-yv)))
        history.append({'epoch':epoch+1,'loss':total/len(y),'val_loss':val_loss,'val_mae':val_mae})
        if val_mae<best:best=val_mae;torch.save(model.state_dict(),folder/'best.pt')
    model.load_state_dict(torch.load(folder/'best.pt',weights_only=True,map_location=device));model.eval()
    with torch.no_grad():prediction=model(xv).cpu().numpy().reshape(-1)
    curves=pd.DataFrame(history);curves.to_csv(folder/'history.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(curves.epoch,curves.loss,label='Training loss');ax.plot(curves.epoch,curves.val_loss,label='Validation loss');ax.legend();ax.set(xlabel='Epoch',ylabel=config['loss'].upper(),title='PyTorch extra-credit DNN');fig.tight_layout();fig.savefig(folder/'loss.png',dpi=140);plt.close(fig)
    result={'framework':'pytorch','hardware':str(device),**config,'epochs_completed':100,'seconds':time.perf_counter()-start,**metrics(targets['validation'],prediction)}
    (folder/'result.json').write_text(json.dumps(result,indent=2))
    return result


def save_error_plots(actual: np.ndarray, predicted: np.ndarray, folder: Path, prefix: str) -> None:
    """Save actual/predicted and residual plots; any focused axes are explicitly labeled."""
    import matplotlib.pyplot as plt
    actual=np.asarray(actual).reshape(-1);predicted=np.asarray(predicted).reshape(-1)
    residual=predicted-actual
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    limit=max(1.,float(np.quantile(actual,.99)))
    axes[0].scatter(actual,predicted,s=7,alpha=.15)
    axes[0].plot([0,limit],[0,limit],color='black',linestyle='--')
    axes[0].set(xlim=(0,limit),ylim=(0,limit),xlabel='Actual minutes',ylabel='Predicted minutes',title='Focused display: actual-duration 99th percentile')
    axes[1].hist(residual,bins=80)
    axes[1].set(xlabel='Prediction minus actual (minutes)',ylabel='Trips',title='Residuals: all evaluation rows')
    fig.suptitle('Display limits do not remove rows from the error metrics')
    fig.tight_layout();folder.mkdir(parents=True,exist_ok=True);fig.savefig(folder/f'{prefix}_errors.png',dpi=140);plt.close(fig)
