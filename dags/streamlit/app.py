import pickle
import tensorflow

import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tensorflow as tf
from datetime import timedelta

import time

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import keras._tf_keras.keras.initializers
from keras._tf_keras.keras.layers import Dense, Layer, LSTM, GRU, SimpleRNN, RNN
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.regularizers import l1, l2

# Add directory in the path

import sys
import os

from pathlib import Path

# Get the user's home directory
home_dir = Path.home()

# Construct a path relative to the home directory
file_path = home_dir / "dags/streamlit/pages/models/2YR_models_rnn.pkl"
# st.write(file_path)


# from model_functions import SimpleRNN_, GRU_, LSTM_
# # from functions import SimpleRNN_, GRU_, LSTM_

# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True

# Set the page configuration
st.set_page_config(page_title="My Streamlit App", page_icon="🌟")

st.write(f"Running on Tensorflow version: {tensorflow.__version__}")

with open('dags/streamlit/pages/styles.css') as css: 
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

st.markdown("# Welcome to RMD1 Machine Learning Portal!")

st.markdown("Make use of recurrent neural network (RNN), gated recurrent unit (GRU), and long-short term memory (LSTM) for bond yiend time series forecast.")

# ##############################################################

# # Open 2-year models

# models_path = "/pages/models"

with open(f"{file_path}/2YR_models_rnn.pkl", "rb") as file:
     data_2yr_rnn = pickle.load(file)

# with open(f"{model_path}/2YR_models_gru.pkl", "rb") as file:
#      data_2yr_gru = pickle.load(file)

# with open(f"{model_path}/2YR_models_lstm.pkl", "rb") as file:
#      data_2yr_lstm = pickle.load(file)

# data_2yr = {**data_2yr_rnn, **data_2yr_gru, **data_2yr_lstm}

# st.markdown("#### Best parameters for each model")

# st.markdown(""" <style> .center-text { text-align: center; font-weight: bold; font-size: 20px; } </style> """, unsafe_allow_html=True)
# st.markdown('<p class="center-text">2-year UST models:</p>', unsafe_allow_html=True)

# table_models_2yr = pd.DataFrame.from_dict(data_2yr, orient='index')
# table_models_2yr.drop(
#     columns=["model", "function", "label",
#              "pred_train", "pred_test", "pred_train_scaled", "MSE_train", 'MSE_test', 'MSE_val',
#              "pred_test_scaled", "y_train_scaled", "y_test_scaled", "pred_val",
#              "pred_val_scaled", "y_val_scaled",
#           #    'scaler_train_mean', 'scaler_train_scale',
#           #    'scaler_val_mean', 'scaler_val_scale', 'scaler_test_mean', 'scaler_test_scale',
#              'MSE_train_scaled', 'MSE_val_scaled', 'MSE_test_scaled',
#           #    'MAE_train_scaled', 'MAE_test_scaled', 'MAE_val_scaled',
#              'R2_train_scaled', 'R2_test_scaled', 'R2_val_scaled',
#              'scaler_train_mean', 'scaler_train_std',
#              'scaler_test_mean', 'scaler_test_std',
#              'scaler_val_mean', 'scaler_val_std', 'cv_results'
#              ], 
#     inplace=True)
# table_models_2yr['cv_time'] = table_models_2yr['cv_time'].apply(lambda x: (x/60)/60)
# table_models_2yr['train_time'] = table_models_2yr['train_time'].apply(lambda x: x/60)
# table_models_2yr.rename(columns={'cv_time': 'Cross-val time (hrs)',
#                      'train_time': 'Training time (mins)',
#                      'l1_reg': 'L1 Regularization',
#                      'H': 'Nodes',
#                     #  'MSE_train': 'MSE Train',
#                     #  'MSE_test':'MSE Test',
#                     #  'MSE_val': 'MSE Val',
#                      'MAE_train_scaled': 'MAE Train',
#                      'MAE_val_scaled': 'MAE Val',
#                      'MAE_test_scaled': 'MAE Test'}, inplace=True)
# st.table(table_models_2yr)
