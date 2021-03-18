"""Finding line-specific buffer capacity factor"""

from c_linecluster import contingency_factor_cluster
import numpy as np
import pypsa
__author__ = "Amin Shokri Gazafroudi (KIT)"
__copyright__ = f"Copyright 2021, {__author__}, GNU GPL 3"

import warnings
warnings.simplefilter("ignore")

# clustering buses of the system based on buses of each line

# network_name =f"../networks/main_paper_prenetwork_de/prenetwork.nc"
# n = pypsa.Network(network_name)
n = pypsa.Network(snakemake.input[0])

# Find robust buffer capacity factor for each line subset (line-specific buffer capacity factor)

c = contingency_factor_cluster(
    n, 0, 'DE', c_start=0., c_end=1., app='robust', N=1e4)
out1 = pd.DataFrame({'Cluster no.': cluster_list,
                     'C_line': c})
# out1.to_csv('c_l_list.csv')
out1.to_csv(snakemake.output[0])
