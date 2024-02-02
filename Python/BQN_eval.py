import numpy as np
import os
import pandas as pd
import scipy.stats as sps
import properscoring as ps
import json
import sys
import ast
import re

print('\n\n')
print(sys.executable)
try:
    data = pd.read_csv('../Datasets/DE.csv', index_col=0)
except:
    data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)

distribution = 'Normal'
num_runs = 1
quantile_array = np.arange(0.01, 1, 0.01)

def pinball_score(observed, pred_quantiles):
    quantiles = np.arange(0.01, 1, 0.01)
    scores = pd.Series(np.maximum((1 - quantiles) * (pred_quantiles - observed), quantiles * (observed - pred_quantiles)))
    return scores.mean()

quant_dfs = []

#load data
for num in range(num_runs):

    if num_runs == 1:
        file_path = f'/home/ahaas/BachelorThesis/forecasts_probNN_BQN_0'
    else:
        file_path = f'/home/ahaas/BachelorThesis/forecasts_probNN_BQN_{num + 1}'
    dist_file_list = sorted(os.listdir(file_path))
    print(file_path)

    fc_df = pd.DataFrame()
    for day, file in enumerate(dist_file_list):
        with open(os.path.join(file_path, file)) as f:
            fc = pd.read_csv(f, index_col=0)
        fc_df = pd.concat([fc_df, fc], axis=0)

    quant_dfs.append(fc_df.add_suffix(f'_{num+1}'))

data.index = pd.to_datetime(data.index)
for num, df in enumerate(quant_dfs):
    df[f'forecast_quantiles_{num + 1}'] = df[f'forecast_quantiles_{num + 1}'].apply(lambda x: re.sub(r'\[\s+', '[', x))
    df[f'forecast_quantiles_{num + 1}'] = df[f'forecast_quantiles_{num + 1}'].apply(lambda x: x.replace(' ', ','))
    df[f'forecast_quantiles_{num + 1}'] = df[f'forecast_quantiles_{num + 1}'].apply(lambda x: re.sub(',+', ',', x))
    df[f'forecast_quantiles_{num+1}'] = df[f'forecast_quantiles_{num+1}'].apply(ast.literal_eval)
    y = data.loc[df.index, 'Price']
    crps_obs = [pinball_score(obs, np.array(pred)) for obs, pred in zip(y, df[f'forecast_quantiles_{num+1}'])]
    CRPS = np.mean(crps_obs)
    print(f'\n\nCRPS for trial {num+1}: {CRPS}')
