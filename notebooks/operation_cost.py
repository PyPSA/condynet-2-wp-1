"""calculating the operational costs of a network with different buffer capacity factors """

__author__ = "Amin Shokri Gazafroudi (KIT)"
__copyright__ = f"Copyright 2021, {__author__}, GNU GPL 3"


import pypsa
from pypsa.pf import calculate_PTDF
from pypsa.contingency import calculate_BODF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use(["bmh", "matplotlibrc"])

#n_pre = pypsa.Network("../contingency-workflow-2/networks/prenetwork.nc")

# n00 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.0.nc")
# n05 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.05.nc")
# n10 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.1.nc")
# n15 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.15.nc")
# n20 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.2.nc")
# n25 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.25.nc")
# n30 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.3.nc")
# n35 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.35.nc")
# n40 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.4.nc")
# n45 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.45.nc")
# n40 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.4.nc")
# n45 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.45.nc")
# n50 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.5.nc")
# n55 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.55.nc")
# n60 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.6.nc")
# n65 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.65.nc")
# n70 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.7.nc")
# n75 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.75.nc")
# n80 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.8.nc")
# n85 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.85.nc")
# n90 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.9.nc")
# n95 = pypsa.Network("../contingency-workflow-2/results/noESS_co2_25_C_0to100/postnetwork_heur0.95.nc")
n100 = pypsa.Network("../results/noESS_co2_25_C_0to100/postnetwork_heur1.0.nc")
nfull = pypsa.Network("../results/noESS_co2_25_C_0to100/postnetwork_full.nc")


# co2_price = 100

# co2_costs40 = n40.generators.carrier.map(n40.carriers.co2_emissions)*co2_price /n40.generators.efficiency
# n40.generators.marginal_cost += co2_costs40

# co2_costs45 = n45.generators.carrier.map(n45.carriers.co2_emissions)*co2_price /n45.generators.efficiency
# n45.generators.marginal_cost += co2_costs45

# co2_costs50 = n50.generators.carrier.map(n50.carriers.co2_emissions)*co2_price /n50.generators.efficiency
# n50.generators.marginal_cost += co2_costs50

# co2_costs55 = n55.generators.carrier.map(n55.carriers.co2_emissions)*co2_price /n55.generators.efficiency
# n55.generators.marginal_cost += co2_costs55

# co2_costs60 = n60.generators.carrier.map(n60.carriers.co2_emissions)*co2_price /n60.generators.efficiency
# n60.generators.marginal_cost += co2_costs60

# co2_costs65 = n65.generators.carrier.map(n65.carriers.co2_emissions)*co2_price /n65.generators.efficiency
# n65.generators.marginal_cost += co2_costs65

# co2_costs70 = n70.generators.carrier.map(n70.carriers.co2_emissions)*co2_price /n70.generators.efficiency
# n70.generators.marginal_cost += co2_costs70

# co2_costs75 = n75.generators.carrier.map(n75.carriers.co2_emissions)*co2_price /n75.generators.efficiency
# n75.generators.marginal_cost += co2_costs75

# co2_costs80 = n80.generators.carrier.map(n80.carriers.co2_emissions)*co2_price /n80.generators.efficiency
# n80.generators.marginal_cost += co2_costs80

# co2_costs85 = n85.generators.carrier.map(n85.carriers.co2_emissions)*co2_price /n85.generators.efficiency
# n85.generators.marginal_cost += co2_costs85

# co2_costs90 = n90.generators.carrier.map(n90.carriers.co2_emissions)*co2_price /n90.generators.efficiency
# n90.generators.marginal_cost += co2_costs90

# co2_costs95 = n95.generators.carrier.map(n95.carriers.co2_emissions)*co2_price /n95.generators.efficiency
# n95.generators.marginal_cost += co2_costs95

# co2_costsfull = nfull.generators.carrier.map(nfull.carriers.co2_emissions)*co2_price /nfull.generators.efficiency
# nfull.generators.marginal_cost += co2_costsfull
# c00 = (n00.generators_t.p.sum()*n00.generators.sign*n00.generators.marginal_cost).sum()
# c05 = (n05.generators_t.p.sum()*n05.generators.sign*n05.generators.marginal_cost).sum()
# c10 = (n10.generators_t.p.sum()*n10.generators.sign*n10.generators.marginal_cost).sum()
# c15 = (n15.generators_t.p.sum()*n15.generators.sign*n15.generators.marginal_cost).sum()
# c20 = (n20.generators_t.p.sum()*n20.generators.sign*n20.generators.marginal_cost).sum()
# c25 = (n25.generators_t.p.sum()*n25.generators.sign*n25.generators.marginal_cost).sum()
# c30 = (n30.generators_t.p.sum()*n30.generators.sign*n30.generators.marginal_cost).sum()
# c35 = (n35.generators_t.p.sum()*n35.generators.sign*n35.generators.marginal_cost).sum()
# c40 = (n40.generators_t.p.sum()*n40.generators.sign*n40.generators.marginal_cost).sum()
# c45 = (n45.generators_t.p.sum()*n45.generators.sign*n45.generators.marginal_cost).sum()
# c50 = (n50.generators_t.p.sum()*n50.generators.sign*n50.generators.marginal_cost).sum()
# c55 = (n55.generators_t.p.sum()*n55.generators.sign*n55.generators.marginal_cost).sum()
# c60 = (n60.generators_t.p.sum()*n60.generators.sign*n60.generators.marginal_cost).sum()
# c65 = (n65.generators_t.p.sum()*n65.generators.sign*n65.generators.marginal_cost).sum()
# c70 = (n70.generators_t.p.sum()*n70.generators.sign*n70.generators.marginal_cost).sum()
# c75 = (n75.generators_t.p.sum()*n75.generators.sign*n75.generators.marginal_cost).sum()
# c80 = (n80.generators_t.p.sum()*n80.generators.sign*n80.generators.marginal_cost).sum()
# c85 = (n85.generators_t.p.sum()*n85.generators.sign*n85.generators.marginal_cost).sum()
# c90 = (n90.generators_t.p.sum()*n90.generators.sign*n90.generators.marginal_cost).sum()
# c95 = (n95.generators_t.p.sum()*n95.generators.sign*n95.generators.marginal_cost).sum()
c100 = (n100.generators_t.p.sum()*n100.generators.sign *
        n100.generators.marginal_cost).sum()
cfull = (nfull.generators_t.p.sum()*nfull.generators.sign *
         nfull.generators.marginal_cost).sum()


dif = cfull-c100
print('c100=', c100)

print('cfull=', cfull)

print('dif=', dif)
