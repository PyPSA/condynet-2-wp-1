"""
Check power flows for each branch outage given N-0 network with heuristic contingency factor.
Check out https://pypsa.readthedocs.io/en/latest/contingency_analysis.html#linear-power-flow-contingency-analysis
"""

__author__ = "Fabian Neumann (KIT), Amin Shokri Gazafroudi(KIT)"
__copyright__ = f"Copyright 2021, {__author__}, GNU GPL 3"

import pypsa
import pandas as pd

from utils import get_branch_outages

if __name__ == "__main__":

    n = pypsa.Network(snakemake.input[0])

    # The contingency LPF needs set points for the components
    # from the optimization
    # See https://pypsa.org/examples/scigrid-sclopf.html
    n.generators_t.p_set = n.generators_t.p.copy()
    n.storage_units_t.p_set = n.storage_units_t.p.copy()
    n.links_t.p_set = n.links_t.p0.copy()

    outage_flows = {}
    for sn in n.snapshots:
        outage_flows[sn] = n.lpf_contingency(
            snapshots=sn, branch_outages=get_branch_outages(n)
        )

    pd.concat(outage_flows).to_csv(snakemake.output[0])
