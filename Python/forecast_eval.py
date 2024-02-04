import numpy as np
import os
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import json
import re
import ast


start_time = '2019-06-11'
length = 24
draw_lines = False
draw_shades = True

fcs = {'BQN_AllInp': '/home/ahaas/BachelorThesis/forecasts_probNN_BQN2_0',
       'Normal_3': '/home/ahaas/BachelorThesis/distparams_probNN_normal_3'
       }

data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)
data.index = pd.to_datetime(data.index)
quantile_array = np.arange(0.01, 1, 0.01)
timeframe = pd.date_range(pd.to_datetime(start_time), periods=length, freq='H')

fig, axs = plt.subplots(nrows=(len(fcs)), ncols=1, figsize=(16,9), sharey=True)

for idx, (model, filepath) in enumerate(fcs.items()):

    #plot true price
    axs[idx].plot(timeframe, data.loc[timeframe, 'Price'], color='Black', label='True Price')

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
        dist_file_list = sorted(os.listdir(filepath))
        for day, file in enumerate(dist_file_list):
            with open(os.path.join(filepath, file)) as f:
                fc_dict = json.load(f)

            df = pd.DataFrame(fc_dict)
            df.index = pd.date_range(pd.to_datetime(file), periods=len(df), freq='H')

        quantiles = df.apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc'], scale=x[f'scale'], a=x[f'skewness'], b=x[f'tailweight']), axis=1)
        df[f'forecast_quantiles'] = quantiles
        df = df.drop(['loc', 'scale'], axis=1)
    else:
        raise ValueError('ERROR: could not find ModelType')

    #plot median
    df.index = pd.to_datetime(df.index)
    axs[idx].plot(timeframe, df.loc[timeframe, 'forecast_quantiles'].apply(lambda x: x[49]), color='red', linestyle='--', label='median')

    #highlight quantiles by shades
    if draw_shades:
        for q in np.arange(49, step=5):
            axs[idx].fill_between(timeframe, df.loc[timeframe, 'forecast_quantiles'].apply(lambda x: x[q]),
                                  df.loc[timeframe, 'forecast_quantiles'].apply(lambda x: x[98-q]), alpha=(0.1))
    #highlight quantiles by lines
    if draw_lines:
        for q in np.arange(49, step=5):
            axs[idx].plot(timeframe, df.loc[timeframe, 'forecast_quantiles'].apply(lambda x: x[q]), color='blue', alpha=0.1 + (q/100))
            axs[idx].plot(timeframe, df.loc[timeframe, 'forecast_quantiles'].apply(lambda x: x[98-q]), color='blue',alpha=0.1 + (q/100))

    axs[idx].set_title(model)

    #average quantile dist
    avg_dist1 = df['forecast_quantiles'].apply(lambda x: x[-1] - x[0]).mean()
    avg_dist2 = df['forecast_quantiles'].apply(lambda x: x[74] - x[24]).mean()
    avg_dist3 = df['forecast_quantiles'].apply(lambda x: x[-1] - x[74]).mean()
    avg_dist4 = df['forecast_quantiles'].apply(lambda x: x[24] - x[0]).mean()
    print(f'Quantile differences for model: '+ model + 'over the whole forecast period')
    print(f'Range 99% - 1%: {avg_dist1}')
    print(f'Range 75% - 25%: {avg_dist2}')
    print(f'Range 99% - 75%: {avg_dist3}')
    print(f'Range 25% - 1%: {avg_dist4}\n\n')

    avg_dist1 = df.loc[timeframe, 'forecast_quantiles'].apply(lambda x: x[-1] - x[0]).mean()
    avg_dist2 = df.loc[timeframe, 'forecast_quantiles'].apply(lambda x: x[74] - x[24]).mean()
    avg_dist3 = df.loc[timeframe, 'forecast_quantiles'].apply(lambda x: x[-1] - x[74]).mean()
    avg_dist4 = df.loc[timeframe, 'forecast_quantiles'].apply(lambda x: x[24] - x[0]).mean()
    print(f'Quantile differences for model: '+ model + 'over shown forecast period')
    print(f'Range 99% - 1%: {avg_dist1}')
    print(f'Range 75% - 25%: {avg_dist2}')
    print(f'Range 99% - 75%: {avg_dist3}')
    print(f'Range 25% - 1%: {avg_dist4}\n\n')

plt.legend()
plt.show()