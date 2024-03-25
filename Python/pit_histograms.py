import numpy as np
import os
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import json
import re
import ast

#define evaluated models
fcs = {#'BQN_AllInp': '/home/ahaas/BachelorThesis/forecasts_probNN_BQN2_0',
       'Normal_3': '/home/ahaas/BachelorThesis/distparams_probNN_normal_3'
       }

#get data
data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)
data.index = pd.to_datetime(data.index)
quantile_array = np.arange(0.01, 1, 0.01)

def find_bin(value, quantiles):
    for i, q in enumerate(quantiles):
        if value <= q:
            return i
    return len(quantiles)

#get data from models, stored in df
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
    else:
        raise ValueError('ERROR: could not find ModelType')

    bin_counts = np.zeros(100)

    for y, quantiles in zip(data.iloc[:, 0], df[f'forecast_quantiles']):
        bin_index = find_bin(y, quantiles)
        bin_counts[bin_index] += 1

    normalized_bin_counts = bin_counts/sum(bin_counts)
    plt.figure(figsize=(10, 6), dpi=600)
    plt.ylabel('Share of bin')
    plt.xlabel('bins')
    plt.bar(range(100), normalized_bin_counts)
    plt.show()
