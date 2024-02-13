import optuna
import json
import logging
import sys
import os

distribution = 'Normal'

optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))

study_name = f'FINAL_DE_selection_lead_{distribution.lower()}_1'
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

with open(f'/home/ahaas/BachelorThesis/params_trial_{distribution}_lead{1}.json', 'w') as j:
    json.dump(best_trial.params, j)