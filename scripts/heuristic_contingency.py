"""Run N-0 operational problem with heuristic contingency factor.
Check out https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#"""

__author__ = "Amin Shokri Gazafroudi (KIT), Fabian Neumann (KIT)"
__copyright__ = f"Copyright 2021, {__author__}, GNU GPL 3"

# Difference between this code and main code is that we modeled based on 'rolling horizon'.

import pypsa

import pandas as pd
from vresutils.benchmark import memory_logger, timer
from utils import assert_operational_problem

if __name__ == "__main__":

    n = pypsa.Network(snakemake.input[0])

    assert_operational_problem(n)

    n.lines.s_max_pu = float(snakemake.wildcards.s_max_pu)

    solver_options = snakemake.config["solver"]
    solver_name = solver_options.pop("name")

    with memory_logger(filename=snakemake.log.memory) as mem, timer("solving time") as tim:
        group_size = snakemake.config["group_size"]

        for i in range(int(len(n.snapshots) / group_size)):
            snapshots = n.snapshots[group_size *
                                    i: group_size * i + group_size]
            n.lopf(snapshots=snapshots,
                   pyomo=False,
                   solver_name=solver_name,
                   solver_options=solver_options,
                   formulation="kirchhoff",
                   )

    pd.Series({"time [sec]": tim.usec/1e6,
               "memory [MB]": mem.mem_usage[0]}).to_csv(snakemake.log.stats)

    n.export_to_netcdf(snakemake.output[0])
