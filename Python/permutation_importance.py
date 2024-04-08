import pandas as pd
import os
import json
from datetime import datetime
import numpy as np
import scipy.stats as sps
import re
import ast

distribution = 'JSU'
baseline_path = '/home/ahaas/BachelorThesis/forecasts_probNN_jsu_q-Ens'
def get_df(path):
    if not os.path.exists(path):
        print(f'Path does not exist: {path}\nReturning')
        return

    dist_file_list = sorted(os.listdir(path))
    dist_params = pd.DataFrame()
    for file in dist_file_list:
        with open(os.path.join(path, file), 'r') as f:
            fc_dict = json.load(f)

        fc_df = pd.DataFrame(fc_dict)
        fc_df.index = pd.date_range(start=file, periods=len(fc_df), freq='H')
        dist_params = pd.concat([dist_params, fc_df])
    return dist_params

quantile_array = np.arange(0.01, 1, 0.01)
def pinball_score(observed, pred_quantiles):
    observed = np.asarray(observed)
    pred_quantiles = np.asarray(pred_quantiles)

    if observed.ndim == 0:
        observed_expanded = observed[np.newaxis]
    else:
        observed_expanded = observed[:, np.newaxis]

    losses = np.maximum((1 - quantile_array) * (pred_quantiles - observed_expanded),
                        quantile_array * (observed_expanded - pred_quantiles))
    return np.mean(losses)

data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)
data.index = pd.to_datetime(data.index)

baseline = pd.read_csv(os.path.join(baseline_path, 'predictions.csv'), index_col=0)
baseline = baseline.rename(columns={'0': 'forecast_quantiles'})
baseline[f'forecast_quantiles'] = baseline[f'forecast_quantiles'].apply(lambda x: re.sub(r'\[\s+', '[', x))
baseline[f'forecast_quantiles'] = baseline[f'forecast_quantiles'].apply(lambda x: x.replace(' ', ','))
baseline[f'forecast_quantiles'] = baseline[f'forecast_quantiles'].apply(lambda x: re.sub(',+', ',', x))
baseline[f'forecast_quantiles'] = baseline[f'forecast_quantiles'].apply(ast.literal_eval)
baseline[f'forecast_quantiles'] = baseline[f'forecast_quantiles'].apply(lambda x: np.array(x))
baseline.index = pd.to_datetime(baseline.index)

y = data.loc[baseline.index, 'Price']

param_dfs = []
for hour in range(1, 7):
    ens_dfs = []
    for trial in range(1, 5):
        path = f'/home/ahaas/BachelorThesis/distparams_probNN_{distribution.lower()}_{trial}_Y_1{hour}'
        param_df = get_df(path)
        ens_dfs.append(param_df)
    param_dfs.append(ens_dfs)


if distribution.lower() == 'jsu':
    #generate ensemble forecasts
    ensembles = np.empty((len(param_dfs), len(y), 99))
    for num, ens in enumerate(param_dfs):

        #gettin quantiles for each ensemble member
        data_3d = np.array([df.to_numpy() for df in ens])
        quantiles_3d = np.empty((*data_3d.shape[:2], len(quantile_array)))
        for i, q in enumerate(quantile_array):
            quantiles_3d[:, :, i] = sps.johnsonsu.ppf(q, loc=data_3d[:, :, 0], scale=data_3d[:, :, 1],
                                                      a=data_3d[:, :, 3], b=data_3d[:, :, 2])
        #averaging quantiles to gain q-ens
        ensembles[num, :, :] = quantiles_3d.mean(axis=0)

    # calculate baseline crps
    crps_baseline = [pinball_score(observed, quantiles_row) for observed, quantiles_row in
                     zip(y, baseline[f'forecast_quantiles'])]
    print(f'Baseline CRPS: {np.mean(crps_baseline)}\n')

    #get crps for each ensemble prediction
    crps_ensembles = np.empty(ensembles.shape[:2])
    for i, ens in enumerate(ensembles):
        crps_ensembles[i, :] = pinball_score(y, ens)
        print(f'Ens {i+1} CRPS: {np.mean(crps_ensembles[i, :])}')
        pmi = (np.mean(crps_ensembles[i, :]) - np.mean(crps_baseline))/ np.mean(crps_baseline)
        print(f'PMI: {pmi}\n')










