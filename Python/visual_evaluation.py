import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

#load data
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

#fuction to plot boxplots for every hour
def violon_per_hour(dist_params):
    data = dist_params['loc'].groupby(dist_params.index.hour).agg(list)
    plt.violinplot(data,
                   showmeans=True,
                   showextrema=True,
                   quantiles=[[.05, .95] for _ in range(len(data))])
    plt.plot(range(1, 25), dist_params['loc'].groupby(dist_params.index.hour).mean(), linewidth= 1)
    plt.show()
    if not os.path.exists('/home/ahaas/BachelorThesis/Plots'):
        os.mkdir('/home/ahaas/BachelorThesis/Plots')
    plt.savefig('/home/ahaas/BachelorThesis/Plots/prob_normal_1.svg')


filepath = '/home/ahaas/BachelorThesis/distparams_probNN_normal_1'
violon_per_hour(load_data(filepath))
