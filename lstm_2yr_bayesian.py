from src.ETL import Extract
from src.ETL import Transform

from src.model_functions import build_model
from src.modeling import Modeling

from keras._tf_keras.keras.callbacks import EarlyStopping
import time

start_time = time.time()

dataset_ = "2yr"
model_ = "lstm"
params = {
    model_: {
        'model': None, # will eventually get a keras model instance after running modeling.cross_val 
        'function': build_model, 'color': 'blue',
        'l1_reg': None, 'H': None, 'label': model_.upper(),
        'cv_results': {"means_": None, 'stds_': None, 'params_': None}
        }
    }
batch_size = 360
max_epochs = 250

forecast_horizon = 4


# Instantiate a Transform class with the given dataset
transform_2yr = Transform(dataset_=dataset_)


# transform splits and reformats the data and saves it in the ./data/staging folder
# already computes for the lag (but the lag for now is hardcoded at 15)
transform_2yr.split_and_reformat(
    train_split=0.7,
    val_split=0.1,
    forecast_horizon=forecast_horizon
    )

param_grid = {
    'n_units': [5,10,15,20], # [5,10,15,20,25,30]
    'l1_reg': [0.1, 0.01, 0.001, 0.001]
    }

es = EarlyStopping(
    monitor='loss',
    mode='min',
    verbose=1,
    patience=100,
    min_delta=1e-7,
    restore_best_weights=True
    )


modeling = Modeling(
    batch_size=batch_size,
    max_epochs=max_epochs,
    n_steps=15,
    forecast_horizon=forecast_horizon
    )

modeling.cross_validate(
    params=params,
    param_grid=param_grid,
    dataset=dataset_,
    es=es
    )

modeling.cross_validate_bayes(
    params=params, 
    es=es, 
    dataset=dataset_
    )

modeling.train(
    es=es, 
    dataset_=dataset_,
    model_ = model_
    )

end_time = time.time()
print(f"Time elapsed on batch size {batch_size} and max epoch of {max_epochs} is {end_time - start_time} seconds or {(end_time - start_time)/60} mins.")
# Time elapsed on batch size 360 and max epoch of 250 is 10312.078974962234 seconds or 171.86798291603725 mins.