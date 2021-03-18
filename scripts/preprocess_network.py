"""Preprocess network."""

__author__ = "Fabian Neumann (KIT), Amin Shokri Gazafroudi (KIT)"
__copyright__ = f"Copyright 2021, {__author__}, GNU GPL 3"

import pypsa
import pandas as pd
import numpy as np


def prepare_network(n):
    """
    Several preprocessing steps including load shedding and
    fixing optimised capacities.
    """

    n.lines.s_max_pu = 1.0

    n.generators.p_nom_extendable = False
    n.storage_units.p_nom_extendable = False
    n.links.p_nom_extendable = False
    n.lines.s_nom_extendable = False

#     n.generators.p_nom = n.generators.p_nom_opt
#     n.storage_units.p_nom = n.storage_units.p_nom_opt
#     n.links.p_nom = n.links.p_nom_opt
#     n.lines.s_nom = n.lines.s_nom_opt

    n.remove("GlobalConstraint", "CO2Limit")

    co2_price = 25

    co2_costs = n.generators.carrier.map(
        n.carriers.co2_emissions)*co2_price/n.generators.efficiency

    n.generators.marginal_cost += co2_costs

    if snakemake.config["load_shedding"] and "load" not in n.carriers.index:
        n.add("Carrier", "load")
        n.madd(
            "Generator",
            n.buses.index,
            " load",
            bus=n.buses.index,
            carrier="load",
            sign=1e-3,  # measure p and p_nom in kW
            marginal_cost=1e2,  # Eur/kWh
            p_nom=1e9,  # kW
        )

    return n


def split_outage_lines(n):
    """
    Separate one parallel line in each corridor for which we consider the outage.
    Special handling for 220kV lines lifted to 380kV.
    """

    def create_outage_lines(n, condition, num_parallel):
        lines_out = n.lines.loc[condition].copy()
        lines_out.s_nom = lines_out.s_nom * num_parallel / lines_out.num_parallel
        lines_out.num_parallel = num_parallel
        lines_out.index = [f"{i}_outage" for i in lines_out.index]
        return lines_out

    def adjust_nonoutage_lines(n, condition, num_parallel):
        nump = n.lines.loc[condition, "num_parallel"]
        n.lines.loc[condition, "s_nom"] *= (nump - num_parallel) / nump
        n.lines.loc[condition, "num_parallel"] -= num_parallel

    cond_220 = (n.lines.num_parallel > 0.5) & (n.lines.num_parallel < 1)
    cond_380 = n.lines.num_parallel > 1

    lines_out_220 = create_outage_lines(n, cond_220, 1 / 3)
    lines_out_380 = create_outage_lines(n, cond_380, 1.0)

    adjust_nonoutage_lines(n, cond_220, 1 / 3)
    adjust_nonoutage_lines(n, cond_380, 1)

    n.lines = pd.concat([n.lines, lines_out_220, lines_out_380])

    n.calculate_dependent_values()

    return n


# def apply_hacks(n):
#     """
#     Here's some space to do some hacking to a specific network.
#     """

#     to_remove = n.lines.loc[n.lines.s_nom_opt < 10].index
#     print("Removing following lines with small capacity:",to_remove)
#     n.mremove("Line", to_remove)

#     to_remove = n.lines.index[n.lines.x == np.inf]
#     print("Removing following lines with infinite reactance:",to_remove)
#     n.mremove("Line", to_remove)

#     bus_to_remove = "DE0 62"
#     n.remove("Bus", bus_to_remove)
#     n.mremove("StorageUnit", n.storage_units.loc[n.storage_units.bus == bus_to_remove].index)
#     n.mremove("Generator", n.generators.loc[n.generators.bus == bus_to_remove].index)
#     n.mremove("Load", n.loads.loc[n.loads.bus == bus_to_remove].index)

#     return n


if __name__ == "__main__":

    n = pypsa.Network(snakemake.input[0])

    n = prepare_network(n)

#    n = apply_hacks(n)

    n = split_outage_lines(n)

    n.lines = n.lines.loc[n.lines.num_parallel != 0]

    n.mremove("StorageUnit", n.storage_units.index)

    n.determine_network_topology()

    n.export_to_netcdf(snakemake.output[0])
