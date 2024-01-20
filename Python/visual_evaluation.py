import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

#load data
data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)
data = data.iloc[:, 0]
data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')
plt_data = data.groupby(data.index.hour).agg(list)
plt.violinplot(plt_data,
               showmeans=True,
                showextrema=True,
                quantiles=[[.05, .95] for _ in range(len(plt_data))])
plt.plot(range(1,25), plt_data.apply(np.mean), linewidth=1)
plt.show()

#load params
filepath = '/home/ahaas/BachelorThesis/distparams_leadNN_normal_4'
#load data
def load_data(filepath):
    dist_params = pd.DataFrame()
    for file in sorted(os.listdir(filepath)):
        with open(os.path.join(filepath, file)) as f:
            fc_dict = json.load(f)

        fc_df = pd.DataFrame(fc_dict)
        fc_df.index = pd.date_range(pd.to_datetime(file), periods=len(fc_df), freq='H')
        dist_params = pd.concat([dist_params, fc_df])
    return dist_params

#function to plot boxplots for every hour
def violon_per_hour(dist_params, parameter):
    data = dist_params[parameter].groupby(dist_params.index.hour).agg(list)
    plt.violinplot(data,
                   showmeans=True,
                   showextrema=True,
                   quantiles=[[.05, .95] for _ in range(len(data))])
    plt.plot(range(1, 25), dist_params[parameter].groupby(dist_params.index.hour).mean(), linewidth= 1)
    plt.show()
    #if not os.path.exists('/home/ahaas/BachelorThesis/Plots'):
    #    os.mkdir('/home/ahaas/BachelorThesis/Plots')
    #plt.savefig('/home/ahaas/BachelorThesis/Plots/prob_normal_1.svg')


violon_per_hour(load_data(filepath), 'loc')
