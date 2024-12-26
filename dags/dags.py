from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta

from src.ETL import Extract, Transform, Load
from src.modeling import Modeling

from src.model_functions import build_model

import os
from skopt import BayesSearchCV
# os.environ[] = None

default_args = {
    'retries': 5,
    'email': ['aculanar@bsp.gov.ph'],
    'retry_delay': timedelta(minutes=2),
    'email_on_failure': True,
    'email_on_retry': False
}

def ETL_process(dataset, train_split, val_split, forecast_horizon):

    # Instantiate a Transform class

    transform = Transform(dataset_=dataset)

    transform.split_and_reformat(train_split=train_split,
                                 val_split=val_split,
                                 forecast_horizon=forecast_horizon)

def crossval_model(es, params, dataset_, model, batch_size, max_epochs, n_steps, forecast_horizon):

    # Create a dictionary
    params_ = {
        model: {
        'model': None,
        'function': build_model, 'color': 'blue',
        'l1_reg': None, 'H': None, 'label': model.upper(),
        'cv_results': {"means_": None, 'stds_': None, 'params_': None}
        }
    }

    # Instantiate a modeling class
    modeling = Modeling(
        batch_size=batch_size,
        max_epochs=max_epochs,
        n_steps=n_steps,
        forecast_horizon=forecast_horizon
        )
    
    # modeling.cross_validate(
    #     params=params,
    #     param_grid=params_grid,
    #     es=es,
    #     dataset=dataset_
    #     )
    
    modeling.cross_validate_bayesian(
        params=params,
        es=es,
        dataset=dataset_
        )

def task_train(dataset_, model_):
    pass

with DAG(
    dag_id = 'forecasting_framework',
    description = 'gets data and transforms', 
    start_date = datetime(year=2024, month=12, day=23),
    schedule_interval='@daily'
) as dag:
    
    ########## ETL section

    task_ETL_2yr = PythonOperator(
        task_id = 'task_ETL_2yr',
        python_callable=ETL_process,
        op_kwargs={'dataset': '2yr', 'train_split': None, 'val_split': None, "forecast_horizon": None}
    )
    task_ETL10yr = PythonOperator(
        task_id = 'task_ETL_2yr',
        python_callable=ETL_process,
        op_kwargs={'dataset': '10yr', 'train_split': None, 'val_split': None, "forecast_horizon": None}
    )

    ########## Model selection

    # Cross-validation
    task_cv_2yr_rnn = PythonOperator()
    task_cv_2yr_gru = PythonOperator()
    task_cv_2yr_lstm = PythonOperator()

    task_cv_10yr_rnn = PythonOperator()
    task_cv_10yr_gru = PythonOperator()
    task_cv_2yr_lstm = PythonOperator()

    # Training
    task_train_2yr_rnn = PythonOperator()
    task_train_2yr_gru = PythonOperator()
    task_train_2yr_lstm = PythonOperator()

    task_train_10yr_rnn = PythonOperator()
    task_train_10yr_gru = PythonOperator()
    task_train_10yr_lstm = PythonOperator()

    ########## Sequence of tasks for Apache Airflow to execute
    task_ETL_2yr, task_ETL10yr >> task_cv_2yr_rnn, task_cv_2yr_gru, task_cv_2yr_lstm >> task_cv_10yr_rnn, task_cv_10yr_gru, task_train_10yr_lstm >> task_train_2yr_rnn, task_train_2yr_gru, task_train_2yr_lstm, task_train_10yr_rnn, task_train_10yr_gru, task_train_10yr_lstm

