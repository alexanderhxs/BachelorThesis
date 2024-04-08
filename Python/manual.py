import optuna
import json
import logging
import sys
import os
import pandas as pd
import sqlite3
import scipy
'''
distribution = 'JSU'
trial = 1
optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))

study_name = f'FINAL_DE_selection_single_{distribution.lower()}_{trial}'
storage_directory = '/home/ahaas/BachelorThesis/trialfiles'
storage_name = f'sqlite:///{os.path.join(storage_directory, f"{study_name}.db")}'

if os.path.exists(storage_directory):
    print("Der Dateipfad existiert.")
else:
    print("Der Dateipfad existiert nicht.")

study = optuna.load_study(study_name=study_name, storage=storage_name)

best_trial = study.best_trial
print('Best trial number:', best_trial.number)
print('Value (Metrik):', best_trial.value)
print('Params: ')
for key, value in best_trial.params.items():
    print(f'  {key}: {value}')

with open(f'/home/ahaas/BachelorThesis/params_trial_single_{trial}.json', 'w') as j:
    json.dump(best_trial.params, j)
'''

optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
study_name = f'FINAL_DE_selection_single_jsu_1'
storage_directory = '/home/ahaas/BachelorThesis/trialfiles'

os.makedirs(storage_directory, exist_ok=True)
db_file_path = os.path.join(storage_directory, f'{study_name}.db')

storage_name = f'sqlite:///{os.path.join(storage_directory, f"{study_name}.db")}'

if not os.path.isfile(db_file_path):
    conn = sqlite3.connect(db_file_path)
    conn.close()

study = optuna.load_study(study_name=study_name, storage=storage_name)
try:
    print(f'Trials: {len(study.trials)}')
    trials_with_value = [trial for trial in study.trials if trial.value is not None]
    print(f'Trials with a value: {len(trials_with_value)}')

    if trials_with_value:
        best_trial = min(trials_with_value, key=lambda trial: trial.value)
        best_params = best_trial.params
        best_score = best_trial.value
        print(f"Best parameters: {best_params}, \nBest score: {best_score}")
except:
    print('Study does not exist or encountered an error.')

data = pd.read_csv(f'/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)
data.index = pd.to_datetime(data.index)



