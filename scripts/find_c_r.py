"""Finding robust buffer capacity factor"""

from c_randomcluster import contingency_factor_cluster
import pandas as pd
import pypsa

__author__ = "Amin Shokri Gazafroudi (KIT)"
__copyright__ = f"Copyright 2021, {__author__}, GNU GPL 3"


import warnings

warnings.simplefilter("ignore")


# clustering based on kmeans algorithm
# from c_kmeanscluster import contingency_factor_cluster

# clustering randomly


# network_name =f"../networks/main_paper_prenetwork_de/prenetwork.nc"
# network_name =f"../networks/elec_s_8.nc"

# n = pypsa.Network(network_name)
n = pypsa.Network(snakemake.input[0])

# Find robust buffer capacity factor for each cluster (subset) of buses

c, cluster_list = contingency_factor_cluster(
    n, 0, "DE", c_start=0.5, c_end=1.0, app="robust", N=1e4
)

out1 = pd.DataFrame({"Cluster no.": cluster_list, "C_robust": c})

# out1.to_csv('c_r_list.csv')
out1.to_csv(snakemake.output[0])
C = min(c)
print(C)
