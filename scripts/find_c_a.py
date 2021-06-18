"""Finding approximate buffer capacity factor"""

__author__ = "Amin Shokri Gazafroudi (KIT)"
__copyright__ = f"Copyright 2021, {__author__}, GNU GPL 3"


import warnings

warnings.simplefilter("ignore")
import pypsa
import pandas as pd

# clustering based on kmeans algorithm
# from c_kmeanscluster import contingency_factor_cluster

# clustering randomly
from c_randomcluster import contingency_factor_cluster


# network_name =f"../networks/prenetwork.nc"
# n = pypsa.Network(network_name)
n = pypsa.Network(snakemake.input[0])

# Find approximate buffer capacity factor for each cluster (subset) of buses

c, cluster_list = contingency_factor_cluster(
    n, 0, "DE", c_start=0.0, c_end=1.0, app="approximate", N=1e4
)
out1 = pd.DataFrame({"Cluster no.": cluster_list, "C_approximate": c})

# out1.to_csv('c_a_list.csv')
out1.to_csv(snakemake.output[0])
