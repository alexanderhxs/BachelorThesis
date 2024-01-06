import os
import json
import numpy as np

with open(f'/home/ahaas/BachelorThesis/params_trial_Normal2.json', 'r') as j:
    params = json.load(j)

print(params)
# Beispiel-Daten
fcs = [(1, {'param1': [10, 20, 30], 'param2': [40, 50, 60]}),
       (2, {'param1': [11, 21, 31], 'param2': [41, 51, 61]})]

# Initialisierung von fc_dict
fc_dict = {}
for param in fcs[0][1]:
    fc_dict[param] = [[] for _ in range(24)]

# Iteration über die Daten und Aktualisierung von fc_dict
for hour, fc in fcs:
    for param, values in fc.items():
        fc_dict[param][hour-1].extend(values)

# Umstrukturierung zu fc_list
fc_list = [{param: values for param, values in fc_dict.items()} for fc_dict in fc_dict.values()]

# Ausgabe von fc_list
print(fc_list)
