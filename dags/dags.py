from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta

from src.ETL import Extract, Transform, Load
from src.modeling import Modeling

default_args = {
    'retries': 5,
    'retry_delay': timedelta(minutes=2)
}

def ETL_data(dataset, train_split, val_split, forecast_horizon):

    # Instantiate a Transform class

    transform = Transform(dataset_=dataset)

    transform.split_and_reformat(train_split=train_split,
                                 val_split=val_split,
                                 forecast_horizon=forecast_horizon)

    pass

def crossval_model(es, params, params_grid, dataset_, model, batch_size, max_epochs, n_steps, forecast_horizon):

    # Instantiate a modeling class
    modeling = Modeling(
        batch_size=batch_size,
        max_epochs=max_epochs,
        n_steps=n_steps,
        forecast_horizon=forecast_horizon
        )
    
    modeling.cross_validate(
        params=params,
        param_grid=params_grid,
        es=es,
        dataset=dataset_
        )

    pass

def task_train(dataset_, model_):
    pass

with DAG(
    dag_id = 'forecasting_framework',
    description = 'gets data and transforms', 
    start_date = datetime(year=2024, month=12, day=23),
    schedule_interval='@daily'
) as dag:
    
    ########## ETL section

    task_ETL_2yr = PythonOperator()
    task_ETL10yr = PythonOperator()

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
    task_ETL_2yr, task_ETL10yr > task_cv_2yr_rnn, task_cv_2yr_gru, task_cv_2yr_lstm > task_cv_10yr_rnn, task_cv_10yr_gru, task_train_10yr_lstm > task_train_2yr_rnn, task_train_2yr_gru, task_train_2yr_lstm, task_train_10yr_rnn, task_train_10yr_gru, task_train_10yr_lstm

