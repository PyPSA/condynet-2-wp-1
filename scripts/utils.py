"""Utility functions used in multiple scripts."""

__author__ = "Fabian Neumann (KIT), Amin Shokri Gazafroudi (KIT)"
__copyright__ = f"Copyright 2021, {__author__}, GNU GPL 3"


def get_branch_outages(n):
    """Creates list of considered outages"""
    outages = n.lines.index[n.lines.index.str.contains("_outage")].union(
        n.lines.index[~n.lines.index.str.contains("_outage") & ~(n.lines.index + "_outage").isin(n.lines.index)])
    return [("Line", o) for o in outages]


def assert_operational_problem(n):
    """Checks whether optimisation problem still includes investment variables."""
    for c in n.iterate_components({"Generator", "StorageUnit", "Link", "Line"}):
        attr = "s" if c.name == "Line" else "p"
        extendable = c.df[f"{attr}_nom_extendable"].any()
        assert (
            not extendable
        ), "Problem is not operational. There are extendable components."
