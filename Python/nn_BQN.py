import pandas as pd
import numpy as np
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp  
from tensorflow_probability import distributions as tfd
from datetime import datetime, timedelta
import tensorflow.compat.v2.keras as keras
import logging
import sys
import os
import optuna
from keras import backend as K
from multiprocessing import Pool
import json
import scipy.stats as sps

# Accepts arguments:
#     cty (currently only DE), default: DE
#     distribution (Normal, StudentT, JSU, SinhArcsinh and NormalInverseGaussian), default: Normal

print('\n')
print(sys.executable)
distribution = 'Normal'

trial = 3
d_degree = 12

params = {'Coal': True,
          'Dummy': True,
          'EUA': True,
          'Gas': True,
          'Oil': True,
          'RES_D': True,
          'RES_D-1': True,
          'activation_1': 'softplus',
          'activation_2': 'softplus',
          'dropout': True,
          'dropout_rate': 0.0001,
          'learning_rate': 0.0001,
          'load_D': True,
          'load_D-1': True,
          'load_D-7': False,
          'neurons_1': 1024,
          'neurons_2': 1024,
          'price_D-1': True,
          'price_D-2': True,
          'price_D-3': False,
          'price_D-7': True,
          'regularize_h1_activation': False,
          'regularize_h1_kernel': False,
          'regularize_h2_activation': False,
          'regularize_h2_kernel': False}

if not os.path.exists(f'/home/ahaas/BachelorThesis/forecasts_probNN_BQN_{trial}'):
    os.mkdir(f'/home/ahaas/BachelorThesis/forecasts_probNN_BQN_{trial}')

# read data file
try:
    data = pd.read_csv('../Datasets/DE.csv', index_col=0)
except:
    data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)

data.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S') for e in data.index]

q_level_loss = K.constant(np.arange(0.01, 1, 0.01))
B = np.zeros((d_degree+1, 99))
for d in range(d_degree+1):
    B[d, :] = sps.binom.pmf(d, d_degree, q_level_loss)

def qt_loss(y_true, y_pred):
    # Quantiles calculated via basis and increments
    B_tensor = K.constant(B, shape=(d_degree+1, 99), dtype=tf.float32)
    #y_true = tf.cast(y_true, dtype=tf.float32)
    #y_pred = tf.cast(y_pred, dtype=tf.float32)
    #y_pred = tf.reshape(y_pred, (y_true.shape[0], d_degree+1))

    q = K.dot(K.cumsum(y_pred, axis=1), B_tensor)
    # Calculate CRPS
    e1 = (y_true - q) * tf.constant(q_level_loss, shape=(1, 99), dtype=tf.float32)
    e2 = (q - y_true) * tf.constant(q_level_loss, shape=(1, 99), dtype=tf.float32)

    # Find correct values (max) and return mean
    return tf.reduce_mean(tf.maximum(e1, e2), axis=1)
#print(qt_loss(tf.constant(np.arange(1), dtype=tf.float32), tf.constant(np.random.rand(9), shape=(1,9), dtype=tf.float32)))

def bern_quants(alpha):
    return np.dot(np.cumsum(alpha, axis=0), B)
#qt_loss(tf.constant(1, dtype=tf.float32), tf.constant(np.random.rand(9), dtype=tf.float32, shape= [1,9]))
def runoneday(inp):
    params, dayno = inp
    df = data.iloc[dayno * 24:dayno * 24 + 1456 * 24 + 24]
    #if datetime.strftime(df.index[-24], '%Y-%m-%d') in os.listdir(f'/home/ahaas/BachelorThesis/forecasts_probNN_BQN_{trial}'):
    #    return
    # prepare the input/output dataframes
    Y = df.iloc[:, 0].to_numpy()
    Y = Y[7 * 24:(1456 * 24)]  # skip first 7 days
    fc_period = int(24)
    X = np.zeros(((1456 * 24) + fc_period, 15))
    for d in range(7 * 24, (1456 * 24) + fc_period):
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
        X[d, 13] = df.index[d].weekday()
        X[d, 14] = df.index[d].hour  # lead time
    # '''
    # input feature selection
    colmask = [False] * 15
    if params['price_D-1']:
        colmask[0] = True
    if params['price_D-2']:
        colmask[1] = True
    if params['price_D-3']:
        colmask[2] = True
    if params['price_D-7']:
        colmask[3] = True
    if params['load_D']:
        colmask[4] = True
    if params['load_D-1']:
        colmask[5] = True
    if params['load_D-7']:
        colmask[6] = True
    if params['RES_D']:
        colmask[7] = True
    if params['RES_D-1']:
        colmask[8] = True
    if params['EUA']:
        colmask[9] = True
    if params['Coal']:
        colmask[10] = True
    if params['Gas']:
        colmask[11] = True
    if params['Oil']:
        colmask[12] = True
    if params['Dummy']:
        colmask[13] = True
    colmask[14] = True  # lead

    X = X[:, colmask]
    Xf = X[-fc_period:, :]
    X = X[(7*24):-fc_period, :]

    inputs = keras.Input(X.shape[1])
    last_layer = keras.layers.BatchNormalization()(inputs)
    # dropout
    dropout = params['dropout'] # trial.suggest_categorical('dropout', [True, False])
    if dropout:
        rate = params['dropout_rate'] # trial.suggest_float('dropout_rate', 0, 1)
        drop = keras.layers.Dropout(rate)(last_layer)
        last_layer = drop

    # regularization of 1st hidden layer,
    regularize_h1_activation = params['regularize_h1_activation']
    regularize_h1_kernel = params['regularize_h1_kernel']
    h1_activation_rate = (0.0 if not regularize_h1_activation 
                          else params['h1_activation_rate_l1'])
    h1_kernel_rate = (0.0 if not regularize_h1_kernel 
                      else params['h1_kernel_rate_l1'])
    # define 1st hidden layer with regularization
    hidden = keras.layers.Dense(params['neurons_1'], 
                                activation=params['activation_1'],
                                # kernel_initializer='ones',
                                kernel_regularizer=keras.regularizers.L1(h1_kernel_rate),
                                activity_regularizer=keras.regularizers.L1(h1_activation_rate))(last_layer)
    # regularization of 2nd hidden layer, 
    #activation - output, kernel - weights/parameters of input
    regularize_h2_activation = params['regularize_h2_activation']
    regularize_h2_kernel = params['regularize_h2_kernel']
    h2_activation_rate = (0.0 if not regularize_h2_activation 
                          else params['h2_activation_rate_l1'])
    h2_kernel_rate = (0.0 if not regularize_h2_kernel 
                      else params['h2_kernel_rate_l1'])
    # define 2nd hidden layer with regularization
    hidden = keras.layers.Dense(params['neurons_2'], 
                                activation=params['activation_2'],
                                # kernel_initializer='ones',
                                kernel_regularizer=keras.regularizers.L1(h2_kernel_rate),
                                activity_regularizer=keras.regularizers.L1(h2_activation_rate))(hidden)

    outputs = keras.layers.Dense((d_degree+1), activation='softplus')(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(params['learning_rate']),
                  loss=qt_loss,
                  metrics=[qt_loss])

    X = X[-1500:, :]
    Y = Y[-1500:]
    # define callbacks
    callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
    perm = np.random.permutation(np.arange(X.shape[0]))
    VAL_DATA = .2
    trainsubset = perm[:int((1 - VAL_DATA)*len(perm))]
    valsubset = perm[int((1 - VAL_DATA)*len(perm)):]
    hist = model.fit(X[trainsubset], Y[trainsubset], epochs=1500, validation_data=(X[valsubset], Y[valsubset]), callbacks=callbacks, batch_size=32, verbose=False)


    predDF = pd.DataFrame(index=df.index[-24:])
    predDF['forecast_quantiles'] = pd.NA
    pred = model.predict(Xf)
    predDF.loc[predDF.index[:], 'forecast_quantiles'] = [bern_quants(hour) for hour in pred]
    predDF.to_csv(os.path.join(f'/home/ahaas/BachelorThesis/forecasts_probNN_BQN_{trial}', datetime.strftime(df.index[-24], '%Y-%m-%d')))
    print(datetime.strftime(df.index[-24], '%Y-%m-%d'))
    print(predDF['forecast_quantiles'].apply(lambda x : np.median(x)))
    print(np.mean(hist.history['loss']))
    print(np.mean(hist.history['val_loss']))
    return predDF

#load params
if trial > 0:
    with open(f'/home/ahaas/BachelorThesis/params_trial_{distribution}{trial}.json', 'r') as j:
        params = json.load(j)


inputlist = [(params, day) for day in range(len(data) // 24 - 1456)]
print(len(inputlist))

start_time = datetime.now()
print(f'Program started at {start_time.strftime("%Y-%m-%d %H:%M:%S")}')

#with Pool(8) as p:
#    _ = p.map(runoneday, inputlist)

runoneday(inputlist[0])
#for day in inputlist:
#    runoneday(day)

end_time = datetime.now()
compute_time = (end_time - start_time).total_seconds()
print(f'Program ended at {end_time.strftime("%Y-%m-%d %H:%M:%S")} \n computation time: {str(timedelta(seconds=compute_time))}')