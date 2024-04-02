import numpy as np
import os
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import json
import re
import ast
from datetime import datetime, timedelta
import  random


start_time = '2020-09-24'
length = 24
draw_lines = False
draw_shades = True

fcs = {'q-Ens BQN': '../forecasts_probNN_BQN_q-Ens',
       'q-Ens JSU': '../forecasts_probNN_jsu_q-Ens',
       'q-Ens Normal': '../forecasts_probNN_normal_q-Ens'
       }

start_datetime = datetime.strptime('2019-09-01', "%Y-%m-%d")
end_datetime = datetime.strptime('2020-12-31', "%Y-%m-%d")

delta = end_datetime - start_datetime

random_offset = random.randint(0, delta.days)
random_date = start_datetime + timedelta(days=random_offset)

# Konvertiere das zufällige Datum in einen String und gib es zurück
start_time = random_date.strftime("%Y-%m-%d")
#start_time = '2020-09-24'



data = pd.read_csv('../Datasets/DE.csv', index_col=0)
data.index = pd.to_datetime(data.index)
quantile_array = np.arange(0.01, 1, 0.01)
timeframe = pd.date_range(pd.to_datetime(start_time), periods=length, freq='H')

B = np.zeros((12+1, 99))
for d in range(12+1):
    B[d, :] = sps.binom.pmf(d, 12, quantile_array)
def bern_quants(alphas):
    return np.dot(alphas, B)

fig, axs = plt.subplots(nrows=(len(fcs)), ncols=1, figsize=(16,12), sharey=True, dpi=300)

for idx, (model, filepath) in enumerate(fcs.items()):

    #plot true price
    axs[idx].plot(timeframe, data.loc[timeframe, 'Price'], color='Black', label='True Price')
    axs[idx].set_xlabel('Time', fontsize=12)
    axs[idx].set_ylabel('DA Price', fontsize=12)
    axs[idx].tick_params(axis='both', which='major', labelsize=12)

    #load data, conditioned on modeltype
    if model.startswith('BQN'):
        dist_file_list = sorted(os.listdir(filepath))
        df = pd.DataFrame()
        for day, file in enumerate(dist_file_list):
            with open(os.path.join(filepath, file)) as f:
                fc = pd.read_csv(f, index_col=0)
            df = pd.concat([df, fc], axis=0)
        df[f'alphas'] = df[f'alphas'].apply(lambda x: re.sub(r'\[\s+', '[', x))
        df[f'alphas'] = df[f'alphas'].apply(lambda x: x.replace(' ', ','))
        df[f'alphas'] = df[f'alphas'].apply(lambda x: re.sub(',+', ',', x))
        df[f'alphas'] = df[f'alphas'].apply(ast.literal_eval)
        df[f'alphas'] = df[f'alphas'].apply(lambda x: np.array(x))
        df[f'forecast_quantiles'] = df[f'alphas'].apply(bern_quants)

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
    elif model.startswith('q-Ens'):
        df = pd.read_csv(os.path.join(filepath, 'predictions.csv'), index_col=0)
        df = df.rename(columns={'0': 'forecast_quantiles'})
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(lambda x: re.sub(r'\[\s+', '[', x))
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(lambda x: x.replace(' ', ','))
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(lambda x: re.sub(',+', ',', x))
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(ast.literal_eval)
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(lambda x: np.array(x))

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

    axs[idx].set_title(model, fontsize=14, fontweight='bold')

    #average quantile dist
    avg_dist1 = df['forecast_quantiles'].apply(lambda x: x[-1] - x[0]).mean()
    avg_dist2 = df['forecast_quantiles'].apply(lambda x: x[74] - x[24]).mean()
    avg_dist3 = df['forecast_quantiles'].apply(lambda x: x[-1] - x[74]).mean()
    avg_dist4 = df['forecast_quantiles'].apply(lambda x: x[24] - x[0]).mean()
    print(f'Quantile differences for model: '+ model + ' over the whole forecast period')
    print(f'Range 99% - 1%: {avg_dist1}')
    print(f'Range 75% - 25%: {avg_dist2}')
    print(f'Range 99% - 75%: {avg_dist3}')
    print(f'Range 25% - 1%: {avg_dist4}\n')

    std_dist1 = df['forecast_quantiles'].apply(lambda x: x[-1] - x[0]).std()
    print(f'Quantile distance variablility over whole forecast period for model {model}: {std_dist1}\n\n')

    avg_dist1 = df.loc[timeframe, 'forecast_quantiles'].apply(lambda x: x[-1] - x[0]).mean()
    avg_dist2 = df.loc[timeframe, 'forecast_quantiles'].apply(lambda x: x[74] - x[24]).mean()
    avg_dist3 = df.loc[timeframe, 'forecast_quantiles'].apply(lambda x: x[-1] - x[74]).mean()
    avg_dist4 = df.loc[timeframe, 'forecast_quantiles'].apply(lambda x: x[24] - x[0]).mean()
    print(f'Quantile differences for model: '+ model + ' over shown forecast period')
    print(f'Range 99% - 1%: {avg_dist1}')
    print(f'Range 75% - 25%: {avg_dist2}')
    print(f'Range 99% - 75%: {avg_dist3}')
    print(f'Range 25% - 1%: {avg_dist4}\n')
    print('-'*100 + '\n')

axs[0].legend()
plt.tight_layout()
plt.show()

print(start_time)