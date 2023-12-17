import numpy as np
import os
import pandas as pd

data = pd.read_csv('../Datasets/DE.csv', index_col=0)

file_path = '../lear_QRA'
file_list = sorted(os.listdir(file_path))
fc_df = pd.DataFrame()

def pinball_score(observed, pred_quantiles):
    quantiles = np.arange(0.01, 1, 0.01)
    scores = pd.Series(np.maximum((1 - quantiles) * (pred_quantiles - observed), quantiles * (observed - pred_quantiles)))
    return scores.mean()

for file in file_list:
    fc = pd.read_csv(os.path.join(file_path, file), header=None)
    fc = fc.transpose()
    fc.index = pd.date_range(pd.to_datetime(file), periods=len(fc), freq='H')
    fc_df = pd.concat([fc_df, fc])
data.index = pd.to_datetime(data.index)
y = data.loc[fc_df.index]
y = y.iloc[:, 0]
crps_observations = fc_df.apply(lambda x: pinball_score(y[x.name], x), axis=1)
median = fc_df.iloc[:, 49]
mean = fc_df.apply(np.mean, axis = 1)

mae = np.abs(y - median).mean()
rmse = np.sqrt(((y - mean) ** 2).mean())

print('CRPS: ' + str(np.mean(crps_observations)))
print('MAE: ' + str(mae))
print('RMSE: ' + str(rmse))

df_samp = pd.read_csv('../Python/lear_forecast.csv', index_col=0)
df_samp = df_samp.loc[:, 'forecast_averaged']
df_samp.index = pd.to_datetime(df_samp.index)
df_samp = pd.merge(df_samp, y, left_index=True, right_index=True)
print(df_samp.sample(n=20, axis=0))