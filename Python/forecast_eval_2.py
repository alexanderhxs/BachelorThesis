import numpy as np
import os
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import json
import re
import ast


start_time = '2018-12-27'
length = 736*24
draw_lines = False
draw_shades = True

fcs = {'BQN_AllInp': '/home/ahaas/BachelorThesis/forecasts_probNN_BQN2_0',
       'Normal_3': '/home/ahaas/BachelorThesis/distparams_leadNN_normal_3'
       }

data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)
data.index = pd.to_datetime(data.index)
quantile_array = np.arange(0.01, 1, 0.01)
timeframe = pd.date_range(pd.to_datetime(start_time), periods=length, freq='H')
data = data.loc[timeframe, 'Price']

def pinball_score(observed, pred_quantiles):
    quantiles = np.arange(0.01, 1, 0.01)
    scores = pd.Series(np.maximum((1 - quantiles) * (pred_quantiles - observed), quantiles * (observed - pred_quantiles)))
    return scores.mean()

fig, axs = plt.subplots(nrows=(len(fcs)), ncols=1, figsize=(16,9), sharey=True)

for idx, (model, filepath) in enumerate(fcs.items()):

    #load data, conditioned on modeltype
    if model.startswith('BQN'):
        dist_file_list = sorted(os.listdir(filepath))
        df = pd.DataFrame()
        for day, file in enumerate(dist_file_list):
            with open(os.path.join(filepath, file)) as f:
                fc = pd.read_csv(f, index_col=0)
            df = pd.concat([df, fc], axis=0)
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(lambda x: re.sub(r'\[\s+', '[', x))
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(lambda x: x.replace(' ', ','))
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(lambda x: re.sub(',+', ',', x))
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(ast.literal_eval)
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(lambda x: np.array(x, dtype=np.float64))

    elif model.startswith('Normal'):
        df = pd.DataFrame()
        dist_file_list = sorted(os.listdir(filepath))
        for day, file in enumerate(dist_file_list):
            with open(os.path.join(filepath, file)) as f:
                fc_dict = json.load(f)
            fc_df = pd.DataFrame(fc_dict)
            fc_df.index = pd.date_range(pd.to_datetime(file), periods=len(fc_df), freq='H')
            df = pd.concat([df, fc_df], axis=0)

        quantiles = df.apply(lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc'], scale=x[f'scale']), axis=1)
        df[f'forecast_quantiles'] = quantiles
        df = df.drop(['loc', 'scale'], axis=1)

    elif model.startswith('JSU'):
        df = pd.DataFrame()
        dist_file_list = sorted(os.listdir(filepath))
        for day, file in enumerate(dist_file_list):
            with open(os.path.join(filepath, file)) as f:
                fc_dict = json.load(f)
            fc_df = pd.DataFrame(fc_dict)
            fc_df.index = pd.date_range(pd.to_datetime(file), periods=len(fc_df), freq='H')
            df = pd.concat([df, fc_df], axis=0)

        quantiles = df.apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc'], scale=x[f'scale'], a=x[f'skewness'], b=x[f'tailweight']), axis=1)
        df[f'forecast_quantiles'] = quantiles
        df = df.drop(['loc', 'scale'], axis=1)
    else:
        raise ValueError('ERROR: could not find ModelType')
    df.index = pd.to_datetime(df.index)

    crps = [pinball_score(y, pred_quantiles) for y, pred_quantiles in zip(data, df.loc[timeframe, 'forecast_quantiles'])]
    axs[idx].plot(data.index, pd.Series(crps).rolling(24*7).mean(), color='red', label='CRPS ')

    quant_dist = df.loc[timeframe, 'forecast_quantiles'].apply(lambda x: x[-1] - x[0])
    ax_2 = plt.twinx(axs[idx])
    ax_2.plot(data.index, quant_dist.rolling(24*7).mean(), color='black', label='Quantile distance')

    axs[idx].set_title(model)


plt.legend()
plt.show()

