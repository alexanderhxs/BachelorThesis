import numpy as np
import os
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import json
import sys

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

param_dfs = []

#load data
for num in range(num_runs):
    try:
        if num_runs == 1:
            file_path = f'../distparams_probNN_{distribution.lower()}_2'
        else:
            file_path = f'../distparams_obNN_{distribution.lower()}_{num+1}'

        dist_file_list = sorted(os.listdir(file_path))
    except:
        if num_runs == 1:
            file_path = f'/home/ahaas/BachelorThesis/distparams_leadNN_{distribution.lower()}_4'
        else:
            file_path = f'/home/ahaas/BachelorThesis/distparams_probNN_{distribution.lower()}_{num + 1}'
        dist_file_list = sorted(os.listdir(file_path))
        print(file_path)

    dist_params = pd.DataFrame()
    for day, file in enumerate(dist_file_list):
        with open(os.path.join(file_path, file)) as f:
            fc_dict = json.load(f)

        fc_df = pd.DataFrame(fc_dict)
        fc_df.index = pd.date_range(pd.to_datetime(file), periods=len(fc_df), freq='H')
        dist_params = pd.concat([dist_params, fc_df])

    param_dfs.append(dist_params.add_suffix(f'_{num+1}'))
data.index = pd.to_datetime(data.index)

def plotting(y, loc_series, crps_observations, quantiles):
    plt.plot(y.index, pd.Series(crps_observations).rolling(window=24 * 7).mean(), label='CRPS over fc period',
             color='blue', linewidth=1)
    plt.plot(y.index, (np.sqrt((y.values - loc_series) ** 2)).rolling(window=24 * 7).mean(),
             label='RSME over fc period', color='lightgrey', linestyle='--', linewidth=1)
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    lead_crsps = []
    lead_mae = []
    lead_rsme = []
    for lead in range(24):
        quantiles_lead = quantiles[lead::24]
        loc_series_lead = loc_series[lead::24]
        y_lead = y[lead::24]
        median_series = quantiles_lead.apply(lambda x: x[50])
        crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in
                             zip(y_lead, quantiles_lead)]
        mae = np.abs(y_lead.values - median_series).mean()
        rmse = np.sqrt((y_lead.values - loc_series_lead) ** 2).mean()
        print(f'Results for lead time: {lead}')
        print('Observations: ' + str(len(y)) + '\n')
        print('MAE: ' + str(mae) + '\n' + 'RMSE: ' + str(rmse))
        print('CRPS: ' + str(np.mean(crps_observations)) + '\n\n')
        lead_mae.append(mae)
        lead_rsme.append(rmse)
        lead_crsps.append(np.mean(crps_observations))

    hours = np.arange(24) + 1
    bar_width = 0.3
    bar_offset = np.arange(24) * (bar_width + 0.6)

    plt.bar(bar_offset, lead_mae, width=bar_width, label='MAE', color='lightgrey')
    plt.bar(bar_offset + 0.2, lead_rsme, width=bar_width, label='RSME', color='grey')
    plt.bar(bar_offset + 0.4, lead_crsps, width=bar_width, label='CRPS', color='blue')

    plt.xlabel('Hours')
    plt.ylabel('Values')
    plt.title('Evaluation metrics by lead time')
    plt.legend()
    plt.xticks(bar_offset, hours)
    plt.show()




if distribution.lower() == 'normal':
    for num, df in enumerate(param_dfs):
        num += 1
        quantiles = df.apply(lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
        median_series = df.apply(lambda x: sps.norm.median(loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
        y = data.loc[df.index, 'Price']
        crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, quantiles)]
        mae = np.abs(y.values - median_series).mean()
        rmse = np.sqrt((y.values - df[f'loc_{num}']) ** 2).mean()
        print(f'Overall results for run Nr {num}')
        print('Observations: ' + str(len(y)) + '\n')
        print('MAE: ' + str(mae) + '\n' + 'RMSE: ' + str(rmse))
        print('CRPS: ' + str(np.mean(crps_observations)) + '\n\n')

        plotting(y, df[f'loc_{num}'], crps_observations, quantiles)

    #q-Ens averaging (horizontal) via quantile averaging
    quantiles_runs = param_dfs[0].apply(
        lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc_1'], scale=x[f'scale_1']), axis=1)
    quantiles_runs = pd.DataFrame({'run_1': quantiles_runs})
    loc_runs = pd.DataFrame(param_dfs[0]['loc_1'])

    for num, df in enumerate(param_dfs[1:], start=2):
        qs = df.apply(lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
        qs = pd.DataFrame({f'run_{num}': qs})
        quantiles_runs = pd.merge(quantiles_runs, qs, left_index=True, right_index=True, how='inner')
        loc_runs = pd.merge(loc_runs, df[f'loc_{num}'], left_index=True, right_index=True, how='inner')

    qEns_quantiles = quantiles_runs.apply(np.mean, axis=1)
    qEns_crps_observations2 = [pinball_score(observed, quantiles_row) for observed, quantiles_row in
                               zip(y, qEns_quantiles)]
    median_series = qEns_quantiles.apply(lambda x: x[50])
    loc_series = loc_runs.apply(np.mean, axis=1)

    qEns_mae2 = np.abs(y.values - median_series).mean()
    qEns_rmse2 = np.sqrt(((y - loc_series) ** 2).mean())

    print('q-Ens MAE: ' + str(qEns_mae2))
    print('q-Ens RSME:' + str(qEns_rmse2))
    print('q-Ens CRPS: ' + str(np.mean(qEns_crps_observations2)))

    plotting(y, loc_series, qEns_crps_observations2, qEns_quantiles)

    # p-Ens averaging (vertical) via sampling
    for sample_size in [50, 100, 250, 500, 1000, 2500]:
        samples = param_dfs[0].apply(lambda x: sps.norm.rvs(size=250, loc=x['loc_1'], scale=x['scale_1']), axis=1)
        for num, df in enumerate(param_dfs[1:], start=2):
            sample = df.apply(lambda x: sps.norm.rvs(size=100, loc=x[f'loc_{num}'], scale=x[f'scale_{num}']),
                              axis=1)
            samples = pd.concat([samples, sample], join='inner', axis=1)

        samples.columns = [str(i) for i in range(len(samples.columns))]
        samples = samples.apply(np.concatenate, axis=1)
        pEns_quantiles = samples.apply(lambda x: np.percentile(x, quantile_array * 100))
        pEns_crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in
                                  zip(y, pEns_quantiles)]
        median_series = pEns_quantiles.apply(lambda x: x[50])
        mean_series = samples.apply(np.mean)

        pEns_mae = np.abs(y.values - median_series).mean()
        pEns_rmse = np.sqrt(((y.values - mean_series) ** 2).mean())

        print(f'Results based on sample size of {sample_size} per distribution')
        print(f'p-Ens MAE: {pEns_mae}')
        print(f'p-Ens RMSE: {pEns_rmse}')
        print(f'p-Ens CRPS: {np.mean(pEns_crps_observations)} \n\n')

        if sample_size == 500:
            plotting(y, mean_series, pEns_crps_observations, pEns_quantiles)


