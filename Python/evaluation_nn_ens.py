import numpy as np
import os
import pandas as pd
import scipy.stats as sps
import properscoring as ps
import json

data = pd.read_csv('../Datasets/DE.csv', index_col=0)

distribution = 'Normal'
num_runs = 4
quantile_array = np.arange(0.01, 1, 0.01)

def pinball_score(observed, pred_quantiles):
    quantiles = np.arange(0.01, 1, 0.01)
    scores = pd.Series(np.maximum((1 - quantiles) * (pred_quantiles - observed), quantiles * (observed - pred_quantiles)))
    return scores.mean()

param_dfs = []

#load data
for num in range(num_runs):
    if num_runs == 1:
        file_path = f'../distparams_probNN_{distribution.lower()}_1'
    else:
        file_path = f'../distparams_probNN_{distribution.lower()}_{num+1}'
    dist_file_list = sorted(os.listdir(file_path))
    dist_params = pd.DataFrame()
    for file in dist_file_list:
        with open(os.path.join(file_path, file)) as f:
            fc_dict = json.load(f)

        fc_df = pd.DataFrame(fc_dict)
        fc_df.index = pd.date_range(pd.to_datetime(file), periods=len(fc_df), freq='H')
        dist_params = pd.concat([dist_params, fc_df])
    param_dfs.append(dist_params.add_suffix(f'_{num+1}'))

data.index = pd.to_datetime(data.index)

if distribution.lower() == 'normal':
    for num, df in enumerate(param_dfs):
        num += 1
        quantiles = df.apply(lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
        median_series = df.apply(lambda x: sps.norm.median(loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
        y = data.loc[df.index, 'Price']
        crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, quantiles)]
        mae = np.abs(y.values - median_series).mean()
        rmse = np.sqrt((y.values - df[f'loc_{num}']) ** 2).mean()
        print(f'Run Nr {num}')
        print('Observations: ' + str(len(y)) + '\n')
        print('MAE: ' + str(mae) + '\n' + 'RMSE: ' + str(rmse))
        print('CRPS: ' + str(np.mean(crps_observations)) + '\n\n')

    #p-Ens averaging (vertical)
    all_df = param_dfs[0]
    for df in param_dfs[1:]:
        all_df = pd.merge(all_df, df, how='inner', left_index=True, right_index=True)

    average_dist_params = pd.DataFrame({'loc': all_df.iloc[:, ::2].mean(axis=1),
                                       'scale': all_df.iloc[:, 1::2].mean(axis=1)},
                                       index=all_df.index)
    y = data.loc[average_dist_params.index, 'Price']
    quantiles = average_dist_params.apply(lambda x: sps.norm.ppf(quantile_array, loc=x['loc'], scale=x['scale']), axis=1)
    median_series = average_dist_params.apply(lambda x: sps.norm.median(loc=x['loc'], scale=x['scale']), axis=1)
    pEns_crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, quantiles)]

    crps_accurate = ps.crps_gaussian(y, mu=average_dist_params['loc'], sig=average_dist_params['scale'])
    pEns_mae = np.abs(y.values - median_series.values).mean()
    pEns_rmse = np.sqrt(((y - average_dist_params['loc'].values) ** 2).mean())

    print('p-Ens MAE: ' + str(pEns_mae) + '\n' + 'p-Ens RMSE: ' + str(pEns_rmse))
    print('p-Ens CRPS: ' + str(np.mean(pEns_crps_observations)))
    print('p-Ens Accurate CRPS: ' + str(np.mean(crps_accurate)) + str('\n\n'))

    #q-Ens averaging (horizontal)
    quantiles_runs = param_dfs[0].apply(lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc_1'], scale=x[f'scale_1']), axis=1)
    quantiles_runs = pd.DataFrame({'run_1': quantiles_runs})
    loc_runs = pd.DataFrame(param_dfs[0]['loc_1'])
    for num, df in enumerate(param_dfs[1:], start=2):
        qs = df.apply(lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
        qs = pd.DataFrame({f'run_{num}': qs})
        quantiles_runs = pd.merge(quantiles_runs, qs, left_index=True, right_index=True, how='inner')
        loc_runs = pd.merge(loc_runs, df[f'loc_{num}'], left_index=True, right_index=True, how='inner')

    ens_quantiles = quantiles_runs.apply(np.mean, axis=1)
    qEns_crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, ens_quantiles)]
    median_series = ens_quantiles.apply(lambda x: x[50])
    loc_series = loc_runs.apply(np.mean, axis=1)

    qEns_mae = np.abs(y.values - median_series).mean()
    qEns_rmse = np.sqrt(((y - loc_series) ** 2).mean())

    print('q-Ens MAE: ' + str(qEns_mae))
    print('q-Ens RSME:' + str(qEns_rmse))
    print('q-Ens CRPS: ' + str(np.mean(qEns_crps_observations)))

elif distribution.lower() == 'jsu':

    for num, df in enumerate(param_dfs):
        num += 1
        quantiles = df.apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1)
        median_series = df.apply(lambda x: sps.johnsonsu.median(loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1)
        y = data.loc[df.index, 'Price']
        crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, quantiles)]
        mae = np.abs(y.values - median_series).mean()
        rmse = np.sqrt((y.values - df[f'loc_{num}']) ** 2).mean()
        print(f'Run Nr {num}')
        print('Observations: ' + str(len(y)) + '\n')
        print('MAE: ' + str(mae) + '\n' + 'RMSE: ' + str(rmse))
        print('CRPS: ' + str(np.mean(crps_observations)) + '\n\n')
    '''    
    datetime_indices = [df.index for df in param_dfs]
    index_set = set(datetime_indices[0]).intersection(*datetime_indices[1:])
    index_set = pd.DatetimeIndex(sorted(index_set))
    pdf_values = []
    for num, df in enumerate(param_dfs):
        num += 1
        lower_percentiles = df.loc[index_set, :].apply(lambda x: sps.johnsonsu.ppf(.01, loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1)
        upper_percentiles = df.loc[index_set, :].apply(lambda x: sps.johnsonsu.ppf(.99, loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1)
        lower = np.min(lower_percentiles)
        upper = np.max(upper_percentiles)
        lin = np.linspace(lower, upper, 10000)

        pdf_values.append(df.loc[index_set, :].apply(lambda x: sps.johnsonsu.pdf(lin, loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1))

    convolution = pdf_values[0].tolist()
    for pdf in pdf_values[1:]:
        convolution = fftconvolve(convolution, pdf.tolist(), mode='same')
    '''
    #p-Ens averaging (vertical)
    '''
    all_df = param_dfs[0]
    for df in param_dfs[1:]:
        all_df = pd.merge(all_df, df, how='inner', left_index=True, right_index=True)

    average_dist_params = pd.DataFrame({'loc': all_df.iloc[:, ::4].mean(axis=1),
                                        'scale': all_df.iloc[:, 1::4].mean(axis=1),
                                        'skewness': all_df.iloc[:, 2::4].mean(axis=1),
                                        'tailweight': all_df.iloc[:, 3::4].mean(axis=1)},
                                       index=all_df.index)
    y = data.loc[average_dist_params.index, 'Price']
    quantiles = average_dist_params.apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x['loc'], scale=x['scale'], a=x['skewness'],
                                                                      b=x['tailweight']), axis=1)
    median_series = average_dist_params.apply(lambda x: sps.johnsonsu.median(loc=x['loc'], scale=x['scale'], a=x['skewness'],
                                                                      b=x['tailweight']), axis=1)
    pEns_crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, quantiles)]

    pEns_mae = np.abs(y.values - median_series.values).mean()
    pEns_rmse = np.sqrt(((y - average_dist_params['loc'].values) ** 2).mean())

    print('p-Ens MAE: ' + str(pEns_mae) + '\n' + 'p-Ens RMSE: ' + str(pEns_rmse))
    print('p-Ens CRPS: ' + str(np.mean(pEns_crps_observations)) + '\n\n')
    '''

    #q-Ens averaging (horizontal)
    quantiles_runs = param_dfs[0].apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc_1'], scale=x[f'scale_1'], a=x[f'skewness_1'], b=x['tailweight_1']),
                                        axis=1)
    quantiles_runs = pd.DataFrame({'run_1': quantiles_runs})
    loc_runs = pd.DataFrame(param_dfs[0]['loc_1'])
    for num, df in enumerate(param_dfs[1:], start=2):
        qs = df.apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}'],a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1)
        qs = pd.DataFrame({f'run_{num}': qs})
        quantiles_runs = pd.merge(quantiles_runs, qs, left_index=True, right_index=True, how='inner')
        loc_runs = pd.merge(loc_runs, df[f'loc_{num}'], left_index=True, right_index=True, how='inner')

    ens_quantiles = quantiles_runs.apply(np.mean, axis=1)
    y = data.loc[ens_quantiles.index, 'Price']
    qEns_crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in
                              zip(y, ens_quantiles)]
    median_series = ens_quantiles.apply(lambda x: x[50])
    loc_series = loc_runs.apply(np.mean, axis=1)

    qEns_mae = np.abs(y.values - median_series).mean()
    qEns_rmse = np.sqrt(((y - loc_series) ** 2).mean())

    print('q-Ens MAE: ' + str(qEns_mae))
    print('q-Ens RSME: ' + str(qEns_rmse))
    print('q-Ens CRPS: ' + str(np.mean(qEns_crps_observations)))
else:
    print('Could not calculate scores: Wrong distribution')