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
import time
from multiprocessing import Pool
import json

# Accepts arguments:
#     cty (currently only DE), default: DE
#     distribution (Normal, StudentT, JSU, SinhArcsinh and NormalInverseGaussian), default: Normal

print('\n')
print(sys.executable)

distribution = 'JSU'
paramcount = {'Normal': 2,
              'StudentT': 3,
              'JSU': 4,
              'SinhArcsinh': 4,
              'NormalInverseGaussian': 4,
              'Point': None,
}
INP_SIZE = 221
# read data file
try:
    data = pd.read_csv('../Datasets/DE.csv', index_col=0)
except:
    data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)

data.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S') for e in data.index]
# data = data.iloc[:4*364*24] # take the first 4 years - 1456 days
df_test = data.iloc[-(554+7)*24:]


#functions to enable loading of models
def custom_johnsonsu_layer(t):
    return tfd.JohnsonSU(
        loc=t[..., :24],
        scale=1e-3 + 3 * tf.math.softplus(t[..., 24:48]),
        tailweight=1 + 3 * tf.math.softplus(t[..., 48:72]),
        skewness=t[..., 72:])

def custom_loss_function(y, rv_y):
    return -rv_y.log_prob(y)

missing_models = []
def runoneday(params, dayno, trial, X_perm):
    df = data.iloc[dayno*24:dayno*24+1456*24+24]

    #set safe_path for results
    safe_path = f'/home/ahaas/BachelorThesis/distparams_probNN_{distribution.lower()}_{trial}_R'
    if not os.path.exists(safe_path):
        os.mkdir(safe_path)

    #check if calculations already been made
    if os.path.exists(os.path.join(safe_path, datetime.strftime(df.index[-24], '%Y-%m-%d'))):
        return

    # prepare the input/output dataframes
    Y = np.zeros((1456, 24))
    # Yf = np.zeros((1, 24)) # no Yf for rolling prediction
    for d in range(1456):
        Y[d, :] = df.loc[df.index[d*24:(d+1)*24], 'Price'].to_numpy()
    Y = Y[7:, :] # skip first 7 days


    X = np.zeros((1456+1, INP_SIZE))
    for d in range(7, 1456+1):
        X[d, :24] = df.loc[df.index[(d-1)*24:(d)*24], 'Price'].to_numpy() # D-1 price
        X[d, 24:48] = df.loc[df.index[(d-2)*24:(d-1)*24], 'Price'].to_numpy() # D-2 price
        X[d, 48:72] = df.loc[df.index[(d-3)*24:(d-2)*24], 'Price'].to_numpy() # D-3 price
        X[d, 72:96] = df.loc[df.index[(d-7)*24:(d-6)*24], 'Price'].to_numpy() # D-7 price
        X[d, 96:120] = df.loc[df.index[(d)*24:(d+1)*24], df.columns[1]].to_numpy() # D load forecast
        X[d, 120:144] = df.loc[df.index[(d-1)*24:(d)*24], df.columns[1]].to_numpy() # D-1 load forecast
        X[d, 144:168] = df.loc[df.index[(d-7)*24:(d-6)*24], df.columns[1]].to_numpy() # D-7 load forecast
        X[d, 168:192] = df.loc[df.index[(d)*24:(d+1)*24], df.columns[2]].to_numpy() # D RES sum forecast
        X[d, 192:216] = df.loc[df.index[(d-1)*24:(d)*24], df.columns[2]].to_numpy() # D-1 RES sum forecast
        X[d, 216] = df.loc[df.index[(d-2)*24:(d-1)*24:24], df.columns[3]].to_numpy() # D-2 EUA
        X[d, 217] = df.loc[df.index[(d-2)*24:(d-1)*24:24], df.columns[4]].to_numpy() # D-2 API2_Coal
        X[d, 218] = df.loc[df.index[(d-2)*24:(d-1)*24:24], df.columns[5]].to_numpy() # D-2 TTF_Gas
        X[d, 219] = df.loc[df.index[(d-2)*24:(d-1)*24:24], data.columns[6]].to_numpy() # D-2 Brent oil
        X[d, 220] = df.index[d*24].weekday()
    # '''
    # input feature selection
    colmask = [False] * INP_SIZE
    if params['price_D-1']:
        colmask[:24] = [True] * 24
    if params['price_D-2']:
        colmask[24:48] = [True] * 24
    if params['price_D-3']:
        colmask[48:72] = [True] * 24
    if params['price_D-7']:
        colmask[72:96] = [True] * 24
    if params['load_D']:
        colmask[96:120] = [True] * 24
    if params['load_D-1']:
        colmask[120:144] = [True] * 24
    if params['load_D-7']:
        colmask[144:168] = [True] * 24
    if params['RES_D']:
        colmask[168:192] = [True] * 24
    if params['RES_D-1']:
        colmask[192:216] = [True] * 24
    if params['EUA']:
        colmask[216] = True
    if params['Coal']:
        colmask[217] = True
    if params['Gas']:
        colmask[218] = True
    if params['Oil']:
        colmask[219] = True
    if params['Dummy']:
        colmask[220] = True
    X = X[:, colmask]
    # '''
    Xf = X_perm[dayno-182, colmask]
    Xf = Xf.reshape(1, -1)
    X = X[7:-1, :]

    # load model if exists
    try:
        model = keras.models.load_model(os.path.join(f'/home/ahaas/BachelorThesis/models/porbNN_{distribution.lower()}_{trial}',
                                    f'{datetime.strftime(df.index[-24], "%Y-%m-%d")}.keras'),
                                        custom_objects={'custom_loss_function': custom_loss_function, 'custom_johnsonsu_layer': custom_johnsonsu_layer},
                                        safe_mode=False)
        print('Model loaded Sucessfully')

    except Exception as e:
        print(f'{e}')
        missing_models.append(f'{datetime.strftime(df.index[-24], "%Y-%m-%d")} for Trial: {trial}')

        print('Building a new one')
        inputs = keras.Input(X.shape[1])
        # batch normalization
        batchnorm = True
        if batchnorm:
            norm = keras.layers.BatchNormalization()(inputs)
            last_layer = norm
        else:
            last_layer = inputs
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
        if paramcount[distribution] is None:
            raise ValueError('Paramcount detected as None')
        else:
            # now define parameter layers with their regularization
            param_layers = []
            param_names = ["loc", "scale", "tailweight", "skewness"]
            for p in range(paramcount[distribution]):
                regularize_param_kernel = params['regularize_'+param_names[p]]
                param_kernel_rate = (0.0 if not regularize_param_kernel
                                     else params[str(param_names[p])+'_rate_l1'])
                param_layers.append(keras.layers.Dense(
                    24, activation='linear', # kernel_initializer='ones',
                    kernel_regularizer=keras.regularizers.L1(param_kernel_rate))(hidden))
            # concatenate the parameter layers to one
            linear = tf.keras.layers.concatenate(param_layers)

            # define outputs
            if distribution == 'Normal':
                outputs = tfp.layers.DistributionLambda(
                        lambda t: tfd.Normal(
                            loc=t[..., :24],
                            scale = 1e-3 + 3 * tf.math.softplus(t[..., 24:])))(linear)
            elif distribution == 'JSU':
                outputs = tfp.layers.DistributionLambda(custom_johnsonsu_layer)(linear)
            else:
                raise ValueError(f'Incorrect distribution {distribution}')
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=keras.optimizers.Adam(params['learning_rate']),
                          loss=custom_loss_function,
                          metrics='mae')
        # '''
        # define callbacks
            callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
            perm = np.random.permutation(np.arange(X.shape[0]))
            VAL_DATA = .2
            trainsubset = perm[:int((1 - VAL_DATA)*len(perm))]
            valsubset = perm[int((1 - VAL_DATA)*len(perm)):]
            model.fit(X[trainsubset], Y[trainsubset], epochs=1500, validation_data=(X[valsubset], Y[valsubset]), callbacks=callbacks, batch_size=32, verbose=False)
            model.save(os.path.join(f'/home/ahaas/BachelorThesis/models/porbNN_{distribution.lower()}_{trial}',
                                    f'{datetime.strftime(df.index[-24], "%Y-%m-%d")}.keras'))
    # metrics = model.evaluate(Xf, Yf) # for point its a list of one [loss, MAE]
    # we optimize the returned value, -1 will always take the model with best MAE

    # pred = model.predict(Xf)[0]
    if paramcount[distribution] is not None:
        dist = model(Xf)
        if distribution == 'Normal':
            getters = {'loc': dist.loc, 'scale': dist.scale}
        elif distribution in {'JSU', 'SinhArcsinh', 'NormalInverseGaussian'}:
            getters = {'loc': dist.loc, 'scale': dist.scale, 
                       'tailweight': dist.tailweight, 'skewness': dist.skewness}
        params = {k: [float(e) for e in v.numpy()[0]] for k, v in getters.items()}
        print(params)
        json.dump(params, open(os.path.join(safe_path, datetime.strftime(df.index[-24], '%Y-%m-%d')), 'w'))
        print(f'{datetime.strftime(df.index[-24], "%Y-%m-%d")}, Trial: {trial}')
    else:
        raise ValueError('Distribution Error')
    return

#load params
params_list = []
for t in range(1, 5):
    with open(f'/home/ahaas/BachelorThesis/params_trial_{distribution}{t}.json', 'r') as j:
        params_list.append(json.load(j))

#generating inputlist
inputlist = [(params_list[trial-1], day, trial) for trial in range(1, 5) for day in range(182, len(data) // 24 - 1456)]


with Pool(8) as p:
    #collecting data for permutation
    X_perm = np.zeros((554 + 7, INP_SIZE))
    for d in range(7, 554 + 7):
        X_perm[d, :24] = df_test.iloc[(d - 1) * 24:(d) * 24, 0].to_numpy()  # D-1 price
        X_perm[d, 24:48] = df_test.iloc[(d - 2) * 24:(d - 1) * 24, 0].to_numpy()  # D-2 price
        X_perm[d, 48:72] = df_test.iloc[(d - 3) * 24:(d - 2) * 24, 0].to_numpy()  # D-3 price
        X_perm[d, 72:96] = df_test.iloc[(d - 7) * 24:(d - 6) * 24, 0].to_numpy()  # D-7 price
        X_perm[d, 96:120] = df_test.iloc[(d) * 24:(d + 1) * 24, 1].to_numpy()  # D load forecast
        X_perm[d, 120:144] = df_test.iloc[(d - 1) * 24:(d) * 24, 1].to_numpy()  # D-1 load forecast
        X_perm[d, 144:168] = df_test.iloc[(d - 7) * 24:(d - 6) * 24, 1].to_numpy()  # D-7 load forecast
        X_perm[d, 168:192] = df_test.iloc[(d) * 24:(d + 1) * 24, 2].to_numpy()  # D RES sum forecast
        X_perm[d, 192:216] = df_test.iloc[(d - 1) * 24:(d) * 24, 2].to_numpy()  # D-1 RES sum forecast
        X_perm[d, 216] = df_test.iloc[(d - 2) * 24:(d - 1) * 24:24, 3].to_numpy()  # D-2 EUA
        X_perm[d, 217] = df_test.iloc[(d - 2) * 24:(d - 1) * 24:24, 4].to_numpy()  # D-2 API2_Coal
        X_perm[d, 218] = df_test.iloc[(d - 2) * 24:(d - 1) * 24:24, 5].to_numpy()  # D-2 TTF_Gas
        X_perm[d, 219] = df_test.iloc[(d - 2) * 24:(d - 1) * 24:24, 6].to_numpy()  # D-2 Brent oil
        X_perm[d, 220] = df_test.index[d].weekday()
    #skip first seven days
    X_perm = X_perm[7:]

    #shuffeling selected rows
    perm_inputs = range(168, 192)
    selected_inputs = X_perm[:, perm_inputs]
    #shuffle columnwise
    for i in range(selected_inputs.shape[1]):
        np.random.shuffle(selected_inputs[:, i])

    X_perm[:, perm_inputs] = selected_inputs
    inputlist = [(params_list[trial - 1], day, trial, X_perm) for trial in range(1, 5) for day in
                 range(182, len(data) // 24 - 1456)]
    _ = p.starmap(runoneday, inputlist)




print(f'Number of Missing Models: {len(missing_models)}')
print('Missing Models:')
for model in missing_models:
    print(model)