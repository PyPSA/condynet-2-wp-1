"""finding total cases and snapshots where the transmission network is overloaded in at least one line for different buffer capacity factors"""

from utils import get_branch_outages

__author__ = "Amin Shokri Gazafroudi (KIT)"
__copyright__ = f"Copyright 2021, {__author__}, GNU GPL 3"


import pypsa
import pandas as pd
import numpy as np

import sys

sys.path.append("../scripts")


c_factor = "1.0"

network_name = f"../results/noESS_co2_25_C_0to100/postnetwork_heur{c_factor}.nc"

n = pypsa.Network(network_name)

filename = f"../results/noESS_co2_25_C_0to100/outage_line_loading_heur{c_factor}.csv"

outages = pd.read_csv(filename, index_col=[0, 1, 2])

X = []
Y = []
loading_list = []
# outage_line_list = []
line_index_list = []
time_list = []

for sn in outages.index.levels[0]:
    outage = outages.loc[sn]
    loading_max = abs(outage.divide(n.passive_branches().s_nom, axis=0)).max()
    loading = abs(outage.divide(n.passive_branches().s_nom, axis=0))

    if loading_max.max() > 1:
        X.append(loading_max.max())
        Y.append(sn)
        for i in range(0, loading.shape[1]):
            for j in range(0, loading.shape[0]):
                if loading.iloc[j, i] > 1:
                    time_list.append(sn)
                    line_index_list.append(loading.index[j])
                    loading_list.append(loading.iloc[j, i])

out1 = pd.DataFrame({"loading_max": np.zeros(len(X))}, index=Y)

for i in range(0, len(Y)):
    out1.loading_max[i] = X[i]

print("Snapshots that overlaoding happened are", len(X), "times")
out1.to_csv("outputs/outage_line_loading_max_heur1.0.csv")

out2 = pd.DataFrame({"Line": line_index_list, "Loading": loading_list}, index=time_list)
print("Total overlaoding happened are", len(loading_list), "times")
out2.to_csv("outputs/list_overload1.0.csv")
