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

distribution = 'Normal'
trial = 4

paramcount = {'Normal': 2,
              'JSU': 4,
              'Point': None}

#select the right folder
if sys.executable != '/home/ahaas/.virtualenvs/BachelorThesis/bin/python':
    folder = '..'
else:
    folder = '/home/ahaas/BachelorThesis'

#load params
with open(f'{folder}/params_trial_{distribution}{trial}.json', 'r') as j:
    params = json.load(j)

#read data
try:
    data = pd.read_csv('../Datasets/DE.csv', index_col=0)
except:
    data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)
data.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S') for e in data.index]

#create directory
if not os.path.exists(f'{folder}/distparams_leadEmbNN_{distribution.lower()}_{trial}'):
    os.mkdir(f'{folder}/distparams_leadEmbNN_{distribution.lower()}_{trial}')

def runoneday(inp):
    params, dayno = inp
    df = data.iloc[dayno*24:dayno*24+1456*24+24]
    # prepare the input/output dataframes
    Y = df.iloc[:, 0].to_numpy()
    Y = Y[7*24:-24] # skip first 7 days

    X = np.zeros(((1456+1)*24, 15))
    for d in range(7*24, (1456*24)+24):
        X[d, 0] = df.iloc[(d-1*24), 0] # D-1 price
        X[d, 1] = df.iloc[(d-2*24), 0] # D-2 price
        X[d, 2] = df.iloc[(d-3*24), 0] # D-3 price
        X[d, 3] = df.iloc[(d-7*24), 0] # D-7 price
        X[d, 4] = df.iloc[d, 1] # D load forecast
        X[d, 5] = df.iloc[(d-1*24), 1] # D-1 load forecast
        X[d, 6] = df.iloc[(d-7*24), 1] # D-7 load forecast
        X[d, 7] = df.iloc[d, 2] # D RES sum forecast
        X[d, 8] = df.iloc[(d-1*24), 2] # D-1 RES sum forecast
        X[d, 9] = df.iloc[(d-2*24), 3] # D-2 EUA
        X[d, 10] = df.iloc[(d-2*24), 4] # D-2 API2_Coal
        X[d, 11] = df.iloc[(d-2*24), 5] # D-2 TTF_Gas
        X[d, 12] = df.iloc[(d-2*24), 6] # D-2 Brent oil
        X[d, 13] = df.index[d].weekday()
        X[d, 14] = df.index[d].hour #lead time
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

    colmask[14] = True
    X = X[:, colmask]
    Xf = X[-24:, :]
    X = X[(7*24):-24, :]

    var_inputs = keras.Input(X.shape[1]-1, name='var_input')
    #var_inputs = keras.layers.BatchNormalization()(var_inputs)

    emb_input = keras.Input(1, name='emb_input')
    lead_emb = keras.layers.Embedding(input_dim=24, output_dim=1, input_length=1)(emb_input)
    lead_emb = keras.layers.Flatten()(lead_emb)

    last_layer = keras.layers.concatenate([var_inputs, lead_emb])

    # dropout
    dropout = params['dropout']  # trial.suggest_categorical('dropout', [True, False])
    if dropout:
        rate = params['dropout_rate']  # trial.suggest_float('dropout_rate', 0, 1)
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
    # activation - output, kernel - weights/parameters of input
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
        outputs = keras.layers.Dense(1, activation='linear')(hidden)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(params['learning_rate']),
                      loss='mae',
                      metrics='mae')
    else:
        # now define parameter layers with their regularization
        param_layers = []
        param_names = ["loc", "scale", "tailweight", "skewness"]
        for p in range(paramcount[distribution]):
            regularize_param_kernel = params['regularize_' + param_names[p]]
            param_kernel_rate = (0.0 if not regularize_param_kernel
                                 else params[str(param_names[p]) + '_rate_l1'])
            param_layers.append(keras.layers.Dense(
                1, activation='linear',  # kernel_initializer='ones',
                kernel_regularizer=keras.regularizers.L1(param_kernel_rate))(hidden))
        # concatenate the parameter layers to one
        linear = tf.keras.layers.concatenate(param_layers)
        # define outputs
        if distribution == 'Normal':
            outputs = tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(
                    loc=t[..., 0],
                    scale=1e-3 + 3 * tf.math.softplus(t[..., 1])))(linear)

        elif distribution == 'JSU':
            outputs = tfp.layers.DistributionLambda(
                lambda t: tfd.JohnsonSU(
                    loc=t[..., 0],
                    scale=1e-3 + 3 * tf.math.softplus(t[..., 1]),
                    tailweight=1 + 3 * tf.math.softplus(t[..., 2]),
                    skewness=t[..., 3]))(linear)
        else:
            raise ValueError(f'Incorrect distribution {distribution}')
        model = keras.Model(inputs=[var_inputs, emb_input], outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(params['learning_rate']),
                      loss=lambda y, rv_y: -rv_y.log_prob(y),
                      metrics='mae')

    #cutting down X to safe fitting time
    #cutter = X.shape[0] * np.random.random_sample(1456-7)
    #X = X[cutter.astype(int), :]
    X = X[-15000:, :]
    Y = Y[-15000:]
    callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
    perm = np.random.permutation(np.arange(X.shape[0]))
    VAL_DATA = .2
    trainsubset = perm[:int((1 - VAL_DATA) * len(perm))]
    valsubset = perm[int((1 - VAL_DATA) * len(perm)):]

    X_train = X[trainsubset]
    X_val = X[valsubset]
    Y_train = Y[trainsubset]
    Y_val = Y[valsubset]

    X_train = {'var_input': X_train[:, :-1],
               'emb_input': X_train[:, -1].astype(int)}
    X_val = {'var_input': X_val[:, :-1],
               'emb_input': X_val[:, -1].astype(int)}
    model.fit(X_train, Y_train, epochs=1500, validation_data=(X_val, Y_val),
              callbacks=callbacks, batch_size=32, verbose=False)

    if paramcount[distribution] is not None:
        Xf = {'var_input': Xf[:, :-1],
              'emb_input': Xf[:, -1].astype(int)}
        dist = model(Xf)
        if distribution == 'Normal':
            getters = {'loc': dist.loc, 'scale': dist.scale}
        elif distribution in {'JSU', 'SinhArcsinh', 'NormalInverseGaussian'}:
            getters = {'loc': dist.loc, 'scale': dist.scale,
                       'tailweight': dist.tailweight, 'skewness': dist.skewness}
        params = {k: v.numpy().tolist() for k, v in getters.items()}
        print(params)
        with open(os.path.join(f'{folder}/distparams_leadNN_{distribution.lower()}_{trial}', datetime.strftime(df.index[-24], '%Y-%m-%d')),'w') as j:
            json.dump(params, j)

inputlist = [(params, day) for day in range(len(data) // 24 - 1456)]
#with Pool(8) as p:
#    _ = p.map(runoneday, inputlist)
for list in inputlist:
    runoneday(list)