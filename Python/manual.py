import os
import json
import numpy as np
import optuna
import sys
import logging

distribution = 'JSU'
trial = 3
optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
study_name = f'FINAL_DE_selection_prob_{distribution.lower()}' # 'on_new_data_no_feature_selection'
storage_name = f'sqlite:////home/ahaas/BachelorThesis/trialfiles/{study_name}{trial}'

study = optuna.load_study(study_name=study_name, storage=storage_name)

with open(f'/home/ahaas/BachelorThesis/params_trial_{distribution}{trial}.json', 'w') as j:
    json.dump(study.best_params, j)


