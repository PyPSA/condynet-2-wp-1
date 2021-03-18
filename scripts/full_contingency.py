"""Run full N-1 constrained operational problem.
Check out https://pypsa.readthedocs.io/en/latest/contingency_analysis.html#security-constrained-linear-optimal-power-flow-sclopf
and https://pypsa.org/examples/scigrid-lopf-then-pf.html for rolling horizon."""

__author__ = "Fabian Neumann (KIT), Amin Shokri Gazafroudi (KIT)"
__copyright__ = f"Copyright 2021, {__author__}, GNU GPL 3"


import pypsa

import pandas as pd
from vresutils.benchmark import memory_logger, timer
from utils import get_branch_outages, assert_operational_problem

if __name__ == "__main__":

    n = pypsa.Network(snakemake.input[0])

    assert_operational_problem(n)

    solver_options = snakemake.config["solver"]

    sclopf_kwargs = {
        "pyomo": False,
        "branch_outages": get_branch_outages(n),
        "solver_name": solver_options.pop("name"),
        "solver_options": solver_options,
        "formulation": "kirchhoff",
    }

    with memory_logger(filename=snakemake.log.memory) as mem, timer("solving time") as tim:
        group_size = snakemake.config["group_size"]
        for i in range(int(len(n.snapshots) / group_size)):
            snapshots = n.snapshots[group_size *
                                    i: group_size * i + group_size]
            n.sclopf(snapshots=snapshots, **sclopf_kwargs)

    pd.Series({"time [sec]": tim.usec/1e6,
               "memory [MB]": mem.mem_usage[0]}).to_csv(snakemake.log.stats)

    n.export_to_netcdf(snakemake.output[0])
