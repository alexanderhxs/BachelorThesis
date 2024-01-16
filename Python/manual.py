import os
import json
from IPython.display import display

import pandas as pd

distribution = 'Normal'

params_list = []

for file in sorted(os.listdir(f'/home/ahaas/BachelorThesis/trials_singleNN_{distribution.lower()}')):
    with open(os.path.join(f'/home/ahaas/BachelorThesis/trials_singleNN_{distribution.lower()}', file), 'r') as j:
        params = json.load(j)
        params_list.append(params)
df = pd.DataFrame(params_list)
df
