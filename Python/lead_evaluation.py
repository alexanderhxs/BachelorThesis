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

distribution = 'JSU'
num_runs = 4
num_runs2 = 4
outlier_threshold = 1000
quantile_array = np.arange(0.01, 1, 0.01)

def pinball_score(observed, pred_quantiles):
    quantiles = np.arange(0.01, 1, 0.01)
    scores = pd.Series(np.maximum((1 - quantiles) * (pred_quantiles - observed), quantiles * (observed - pred_quantiles)))
    return scores.mean()

param_dfs = []
param_dfs2 = []
#load data
for num in range(num_runs):
    if num_runs == 1:
        file_path = f'/home/ahaas/BachelorThesis/distparams_probNN3_{distribution.lower()}_1'
    else:
        file_path = f'/home/ahaas/BachelorThesis/distparams_leadNN3.2_{distribution.lower()}_{num + 1}'
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
for num in range(num_runs2):
    if num_runs2 == 1:
        file_path = f'/home/ahaas/BachelorThesis/distparams_leadNN3.1_{distribution.lower()}_4'
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

    param_dfs2.append(dist_params.add_suffix(f'_{num+1}'))
data.index = pd.to_datetime(data.index)

for num, df in enumerate(param_dfs):
    param_dfs[num] = df.iloc[-24*554:, :]
for num, df in enumerate(param_dfs2):
    param_dfs2[num] = df.iloc[-24*554:, :]
def plotting(y, crps_observations, crps_observations2=None):
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

    # CRPS
    ax1.plot(y[~outliers].index, pd.Series(crps_observations).rolling(window=24 * 7).mean(),
             label='CRPS leadNN-JSU', color='steelblue', linewidth=2)
    if not (crps_observations2 is None):
        ax1.plot(y.index, pd.Series(crps_observations2).rolling(window=24 * 7).mean(),
                 label='CRPS probNN-JSU', color='lightgray', linewidth=2, linestyle = '--')


    # Intraday Variance with scaling for visibility
    intraday_std = y.groupby(y.index.date).std()

    ax2 = ax1.twinx()
    ax2.plot(intraday_std.index, intraday_std.rolling(window=7).mean(),
             label='Scaled Intraday Variance of DA Price', color='goldenrod', linewidth=2)

    # Aesthetics
    plt.grid(axis='y', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Smoothed CRPS value', fontsize=14)
    ax2.set_ylabel('Scaled Intraday Variance', fontsize=14)
    ax1.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.xlabel('Test Period', fontsize=14)
    plt.subplots_adjust(bottom=0.2, top=0.9)

    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower center', fontsize=14, bbox_to_anchor=(0.5, 1.0), ncol=3)

    plt.tight_layout()
    plt.show()

def plot_lead_bar(y, loc_series, quantiles):
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
        #print(f'Results for lead time: {lead}')
        #print('Observations: ' + str(len(y)) + '\n')
        #print('MAE: ' + str(mae) + '\n' + 'RMSE: ' + str(rmse))
        #print('CRPS: ' + str(np.mean(crps_observations)) + '\n\n')
        lead_mae.append(mae)
        lead_rsme.append(rmse)
        lead_crsps.append(np.mean(crps_observations))

    hours = np.arange(24)
    bar_width = 0.3
    bar_offset = np.arange(24) * (1)

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=600)

    ax1.bar(bar_offset - bar_width/2, lead_mae, width=bar_width, label='MAE', color='lightgrey')
    #ax1.bar(bar_offset, lead_rsme, width=bar_width, label='RSME', color='grey')
    ax1.bar(bar_offset + bar_width/2, lead_crsps, width=bar_width, label='CRPS', color='blue')

    ax2 = ax1.twinx()
    ax2.plot(hours, y.groupby(y.index.hour).mean(), label='Mean price', linestyle='--', color='red')
    ax2.plot(hours, y.groupby(y.index.hour).std(), label='Standard deviation price', linestyle='--', color='orange')
    ax2.set_ylim(bottom=0)

    ax1.legend(loc='upper left', fontsize=14)
    ax2.legend(loc='upper right', fontsize=14)
    ax1.set_ylabel('Evaluation metrics', fontsize=14)
    ax2.set_ylabel('DA Price',fontsize=14)
    ax1.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    plt.xticks(fontsize=14)
    plt.xlabel('Hours', fontsize=14)
    #plt.title('Evaluation metrics by lead time')
    plt.show()




if distribution.lower() == 'normal':
    for num, df in enumerate(param_dfs):
        num += 1
        quantiles = df.apply(lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
        median_series = df.apply(lambda x: sps.norm.median(loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
        y = data.loc[df.index, 'Price']
        crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, quantiles)]

        df['crps'] = crps_observations
        df['crps'] = df['crps'].rolling(24*7).mean()
        outlier_mask = df['crps'] > outlier_threshold
        print(df.loc[outlier_mask])

        mae = np.abs(y.values - median_series).mean()
        rmse = np.sqrt((y.values - df[f'loc_{num}']) ** 2).mean()
        print(f'Overall results for run Nr {num}')
        print('Observations: ' + str(len(y)) + '\n')
        print('MAE: ' + str(mae) + '\n' + 'RMSE: ' + str(rmse))
        print('CRPS: ' + str(np.mean(crps_observations)) + '\n\n')

        #plotting(y, df[f'loc_{num}'], crps_observations, quantiles)
        for num2, df2 in enumerate(param_dfs2):
            quantiles2 = df2.apply(lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
            crps_observations2 = [pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, quantiles2)]

            #plotting(y, crps_observations, crps_observations2=crps_observations2)





    if num_runs > 1:
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

        plotting(y, qEns_crps_observations2)
        plot_lead_bar(y, loc_series, qEns_quantiles)

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



if distribution.lower() == 'jsu':

    outliers = []
    for num, df in enumerate(param_dfs):
        num += 1
        quantiles = df.apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1)
        median_series = df.apply(lambda x: sps.johnsonsu.median(loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'],
                                           b=x[f'tailweight_{num}']), axis=1)
        y = data.loc[df.index, 'Price']
        crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, quantiles)]

        #detect outliers
        df['crps'] = crps_observations
        outlier_mask = df['crps'] > outlier_threshold
        outliers.append(outlier_mask)

        no_outlier_df = df.loc[~outlier_mask]
        outlier_df = df.loc[outlier_mask]
        #dd = data[(data.index > '2020-09-01') & (data.index < '2020-10-01')]
        #print(data['Price'].min())
        #print(data['Price'].max())

        mae = np.abs(y.values - median_series).mean()
        rmse = np.sqrt((y.values - df[f'loc_{num}']) ** 2).mean()
        print(f'Overall results for run Nr {num}')
        print('Observations: ' + str(len(y)) + '\n')
        print('MAE: ' + str(mae) + '\n' + 'RMSE: ' + str(rmse))
        print('CRPS: ' + str(np.mean(crps_observations)) + '\n\n')
        print('CRPS without outliers: ' +str(no_outlier_df['crps'].mean()))

        #plotting(y, crps_observations)


    if num_runs > 1:

        # removing all outliers from runs
        outliers = np.any(outliers, axis=0)
        for i, df in enumerate(param_dfs):
            param_dfs[i] = df.loc[~outliers]
        y = y.loc[~outliers]

        #q-Ens averaging (horizontal) via quantile averaging
        quantiles_runs = param_dfs[0].apply(
            lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc_1'], scale=x[f'scale_1'], a=x[f'skewness_1'], b=x[f'tailweight_1']), axis=1)
        quantiles_runs = pd.DataFrame({'run_1': quantiles_runs})
        loc_runs = pd.DataFrame(param_dfs[0]['loc_1'])

        for num, df in enumerate(param_dfs[1:], start=2):
            qs = df.apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1)
            qs = pd.DataFrame({f'run_{num}': qs})
            quantiles_runs = pd.merge(quantiles_runs, qs, left_index=True, right_index=True, how='inner')
            loc_runs = pd.merge(loc_runs, df[f'loc_{num}'], left_index=True, right_index=True, how='inner')

        qEns_quantiles = quantiles_runs.apply(np.mean, axis=1)
        qEns_crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in
                                   zip(y, qEns_quantiles)]
        median_series = qEns_quantiles.apply(lambda x: x[49])
        loc_series = loc_runs.apply(np.mean, axis=1)

        qEns_mae2 = np.abs(y.values - median_series).mean()
        qEns_rmse2 = np.sqrt(((y - loc_series) ** 2).mean())

        print('q-Ens MAE: ' + str(qEns_mae2))
        print('q-Ens RSME:' + str(qEns_rmse2))
        print('q-Ens CRPS: ' + str(np.mean(qEns_crps_observations)))

        bigger_preds = median_series > y
        bigger_preds_probs = bigger_preds.groupby(bigger_preds.index.hour).sum()
        bigger_preds_probs = bigger_preds_probs/ (len(y)//24)
        print(bigger_preds_probs)

        # q-Ens averaging (horizontal) via quantile averaging for 2. model
        quantiles_runs = param_dfs2[0].apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc_1'], scale=x[f'scale_1'], a=x[f'skewness_1'],
                                        b=x[f'tailweight_1']), axis=1)
        quantiles_runs = pd.DataFrame({'run_1': quantiles_runs})
        loc_runs = pd.DataFrame(param_dfs2[0]['loc_1'])

        for num, df in enumerate(param_dfs2[1:], start=2):
            qs = df.apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}'],
                                                      a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1)
            qs = pd.DataFrame({f'run_{num}': qs})
            quantiles_runs = pd.merge(quantiles_runs, qs, left_index=True, right_index=True, how='inner')
            loc_runs = pd.merge(loc_runs, df[f'loc_{num}'], left_index=True, right_index=True, how='inner')
        y = data.loc[df.index, 'Price']
        qEns_quantiles = quantiles_runs.apply(np.mean, axis=1)
        qEns_crps_observations2 = [pinball_score(observed, quantiles_row) for observed, quantiles_row in
                                  zip(y, qEns_quantiles)]
        median_series = qEns_quantiles.apply(lambda x: x[49])
        loc_series = loc_runs.apply(np.mean, axis=1)

        qEns_mae2 = np.abs(y.values - median_series).mean()
        qEns_rmse2 = np.sqrt(((y - loc_series) ** 2).mean())

        print('q-Ens MAE: ' + str(qEns_mae2))
        print('q-Ens RSME:' + str(qEns_rmse2))
        print('q-Ens CRPS: ' + str(np.mean(qEns_crps_observations2)))

        plotting(y, qEns_crps_observations, qEns_crps_observations2)
#        plot_lead_bar(y, loc_series, qEns_quantiles)