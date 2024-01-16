import pandas as pd
import numpy as np
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from datetime import datetime, timedelta
import tensorflow.compat.v2.keras as keras
import sys
import os
from multiprocessing import Pool
import json
import optuna
import logging

paramcount = {'Normal': 2,
              'JSU': 4,
              'Point': None}
distribution = 'Normal'

filepath = f'/home/ahaas/BachelorThesis/trials_singleNN_{distribution.lower()}_2'
if not os.path.exists(filepath):
    os.mkdir(filepath)

#load data
try:
    data = pd.read_csv('../Datasets/DE.csv', index_col=0)
except:
    data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)

binopt = [True, False]
activations = ['relu', 'elu', 'softplus']

#initialization of fcs
fcs = []
df = data.iloc[:, :]
fc_index = pd.to_datetime(df.iloc[(1456*24):].index, format='%Y-%m-%d %H:%M:%S')
fc_index = fc_index[::24]
#train models
def objective(trial, tm):
    # prepare the input/output dataframes
    Y = df.iloc[:, 0].to_numpy()
    Yf = Y[(1456*24)+tm::24]
    Y = Y[(7 * 24)+tm:(1456*24):24]  # skip first 7 days

    X = np.zeros((len(df), 13))
    for d in range((7 * 24)+tm, len(df), 24):
        X[d, 0] = df.iloc[(d - 1 * 24), 0]  # D-1 price
        X[d, 1] = df.iloc[(d - 2 * 24), 0]  # D-2 price
        X[d, 2] = df.iloc[(d - 3 * 24), 0]  # D-3 price
        X[d, 3] = df.iloc[(d - 7 * 24), 0]  # D-7 price
        X[d, 4] = df.iloc[d, 1]  # D load forecast
        X[d, 5] = df.iloc[(d - 1 * 24), 1]  # D-1 load forecast
        X[d, 6] = df.iloc[(d - 7 * 24), 1]  # D-7 load forecast
        X[d, 7] = df.iloc[d, 2]  # D RES sum forecast
        X[d, 8] = df.iloc[(d - 1 * 24), 2]  # D-1 RES sum forecast
        X[d, 9] = df.iloc[(d - 2 * 24), 3]  # D-2 EUA
        X[d, 10] = df.iloc[(d - 2 * 24), 4]  # D-2 API2_Coal
        X[d, 11] = df.iloc[(d - 2 * 24), 5]  # D-2 TTF_Gas
        X[d, 12] = df.iloc[(d - 2 * 24), 6]  # D-2 Brent oil

    #inputfeature selection
    colmask = [False]*13
    if trial.suggest_categorical('price_D-1', [True]):
        colmask[0] = True
    if trial.suggest_categorical('price_D-2', binopt):
        colmask[1] = True
    if trial.suggest_categorical('price_D-3', [True]):
        colmask[2] = True
    if trial.suggest_categorical('price_D-7', [True]):
        colmask[3] = True
    if trial.suggest_categorical('load_D', [True]):
        colmask[4] = True
    if trial.suggest_categorical('load_D-1', binopt):
        colmask[5] = True
    if trial.suggest_categorical('load_D-7', [False]):
        colmask[6] = True
    if trial.suggest_categorical('RES_D', [True]):
        colmask[7] = True
    if trial.suggest_categorical('RES_D-1', [True]):
        colmask[8] = True
    if trial.suggest_categorical('EUA', binopt):
        colmask[9] = True
    if trial.suggest_categorical('Coal', binopt):
        colmask[10] = True
    if trial.suggest_categorical('Gas', binopt):
        colmask[11] = True
    if trial.suggest_categorical('Oil', binopt):
        colmask[12] = True
    X = X[:, colmask]
    Xf = X[(1456*24):, :]
    X = X[(7 * 24):(1456*24), :]
    X = X[tm::24]
    Xf = Xf[tm::24]


    inputs = keras.Input(X.shape[1])
    # batch normalization
    norm = keras.layers.BatchNormalization()(inputs)
    last_layer = norm

    if trial.suggest_categorical('dropout', [False]):
        rate = trial.suggest_float('dropout_rate', 0, 1) # trial.suggest_float('dropout_rate', 0, 1)
        drop = keras.layers.Dropout(rate)(last_layer)
        last_layer = drop

    # regularization of 1st hidden layer,
    regularize_h1_activation = trial.suggest_categorical('regularize_h1_activation',binopt)
    regularize_h1_kernel = trial.suggest_categorical('regularize_h1_kernel', binopt)
    h1_activation_rate = (0.0 if not regularize_h1_activation
                          else trial.suggest_float('h1_activation_rate_l1', 1e-5, 1e1, log=True))
    h1_kernel_rate = (0.0 if not regularize_h1_kernel
                      else trial.suggest_float('h1_activation_rate_l1', 1e-5, 1e1, log=True))
    # define 1st hidden layer with regularization
    hidden = keras.layers.Dense(trial.suggest_int('neurons_1', 32, 200, log=False),
                                activation=trial.suggest_categorical('activation_1', activations),
                                # kernel_initializer='ones',
                                kernel_regularizer=keras.regularizers.L1(h1_kernel_rate),
                                activity_regularizer=keras.regularizers.L1(h1_activation_rate))(last_layer)
    # regularization of 2nd hidden layer,
    #activation - output, kernel - weights/parameters of input
    regularize_h2_activation = trial.suggest_categorical('regularize_h2_activation', binopt)
    regularize_h2_kernel = trial.suggest_categorical('regularize_h2_kernel', binopt)
    h2_activation_rate = (0.0 if not regularize_h2_activation
                          else trial.suggest_float('h2_activation_rate_l1', 1e-5, 1e1, log=True))
    h2_kernel_rate = (0.0 if not regularize_h2_kernel
                      else trial.suggest_float('h2_kernel_rate_l1', 1e-5, 1e1, log=True))
    # define 2nd hidden layer with regularization
    hidden = keras.layers.Dense(trial.suggest_int('neurons_2', 32 , 200, log=False),
                                activation=trial.suggest_categorical('activation_2', activations),
                                # kernel_initializer='ones',
                                kernel_regularizer=keras.regularizers.L1(h2_kernel_rate),
                                activity_regularizer=keras.regularizers.L1(h2_activation_rate))(hidden)

    # now define parameter layers with their regularization
    param_layers = []
    param_names = ["loc", "scale", "tailweight", "skewness"]
    for p in range(paramcount[distribution]):
        regularize_param_kernel = trial.suggest_categorical('regularize_'+param_names[p], binopt)
        param_kernel_rate = (0.0 if not regularize_param_kernel
                             else trial.suggest_float(param_names[p]+'_rate_l1', 1e-5, 1e1, log=True))
        param_layers.append(keras.layers.Dense(
            1, activation='linear', # kernel_initializer='ones',
            kernel_regularizer=keras.regularizers.L1(param_kernel_rate))(hidden))
    # concatenate the parameter layers to one
    linear = tf.keras.layers.concatenate(param_layers)
    # define outputs
    if distribution == 'Normal':
        outputs = tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(
                    loc=t[..., 0],
                    scale = 1e-3 + 3 * tf.math.softplus(t[..., 1])))(linear)
    elif distribution == 'JSU':
        outputs = tfp.layers.DistributionLambda(
                lambda t: tfd.JohnsonSU(
                    loc=t[..., 0],
                    scale=1e-3 + 3 * tf.math.softplus(t[..., 1]),
                    tailweight= 1 + 3 * tf.math.softplus(t[..., 2]),
                    skewness=t[..., 3]))(linear)

    else:
        raise ValueError(f'Incorrect distribution {distribution}')
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)),
                  loss=lambda y, rv_y: -rv_y.log_prob(y),
                  metrics='mae')

    callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
    model.fit(X, Y, epochs=1500, validation_data=(Xf, Yf), callbacks=[optuna.integration.KerasPruningCallback(trial, 'val_loss'), callbacks], batch_size=32, verbose=True)
    metrics = model.evaluate(Xf, Yf)
    return metrics[0]

optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
for model in range(24):
    study_name = f'trial_model_{model}'
    study = optuna.create_study(study_name=study_name, sampler=optuna.samplers.TPESampler())
    study.optimize(lambda trial: objective(trial, model), n_trials=32, n_jobs=8)
    print(study.best_params)
    with open(os.path.join(filepath, study_name), 'w') as j:
        json.dump(study.best_params, j)