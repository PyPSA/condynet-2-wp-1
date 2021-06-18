"""Functions for determining approximate, robust and line-specific contingency factors based on line-specific clustering."""

__author__ = "Amin Shokri Gazafroudi (KIT), Fabian Neumann (KIT), Tom Brown (KIT)"
__copyright__ = f"Copyright 2021, {__author__}, GNU GPL 3"

import numpy as np
import pyomo.kernel as pmo

from pypsa.pf import calculate_PTDF
import random
from numpy.random import rand
import pandas as pd
from chaospy.distributions.sampler.sequences.halton import create_halton_samples

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

import logging

logger = logging.getLogger(__name__)


def find_bus_neighbours(n, bus):
    """
    Find buses in the neighbours.

    Parameters
    ----------
    n : pypsa.Network
    bus : str
        Name of the bus

    Returns
    -------
    neighbours : list
    """
    neighbours = []
    for line in n.lines.index:
        bus0 = n.lines.at[line, "bus0"]
        bus1 = n.lines.at[line, "bus1"]
        if bus0 == bus and bus1 not in neighbours:
            neighbours.append(bus1)
        if bus1 == bus and bus0 not in neighbours:
            neighbours.append(bus0)
    return neighbours


def remove_single_stubs(n):
    """
    Remove buses and lines which are connected to
    only one bus or not connected to any bus.

    Parameters
    ----------
    n : pypsa.Network

    Returns
    -------
    len(stubs) : int
    """
    stubs = []
    for bus in n.buses.index:
        neighbours = find_bus_neighbours(n, bus)
        if len(neighbours) <= 1:
            lines_to_remove = n.lines.index[
                (n.lines.bus0 == bus) ^ (n.lines.bus1 == bus)
            ]
            n.mremove("Line", lines_to_remove)
            stubs.append(bus)

    n.mremove("Bus", stubs)

    return len(stubs)


def remove_stubs(n):
    """
    Remove stubs recursively.

    Parameters
    ----------
    n : pypsa.Network
    """
    while True:
        number_stubs_removed = remove_single_stubs(n)
        if number_stubs_removed == 0:
            return


def calculate_all_PTDF(n):
    """
    Calculate Power Transfer Distribution Factor (PTDF)
    matrices for all subnetworks.

    Parameters
    ----------
    n : pypsa.Network

    Returns
    -------
    None
    """
    n.determine_network_topology()
    n.calculate_dependent_values()

    n.lines["s_nom"] = (
        np.sqrt(3)
        * n.lines.num_parallel
        * n.lines.type.map(n.line_types.i_nom)
        * n.lines.bus0.map(n.buses.v_nom)
    )

    for sub in n.sub_networks.obj:
        calculate_PTDF(sub)


def calculate_s_nom(n, i):
    """
    Find polytope of N-0 secured network.

    Parameters
    ----------
    n : pypsa.Network
    i: int
       number of subnworks (e.g. 0, 1, 2, etc.)

    Returns
    -------
    b : np.ndarray
    s_nom: np.ndarray
    """

    i = int(i)

    n.determine_network_topology()
    n.calculate_dependent_values()

    sub = n.sub_networks.obj[i]

    s_nom = sub.branches().s_nom.values

    b = np.hstack((s_nom, s_nom))

    return b, s_nom


def bounding_box(A, b, solver_name="gurobi"):
    """
    Find upper and lower bounds for each variable.

    Parameters
    ----------
    A : np.ndarray
        e.g. lhs of polytope for N-X secured network
    b : np.ndarray
        e.g. rhs of polytope for N-X secured network
    solver_name : str, default "gurobi"
        Solver

    Returns
    -------
    np.ndarray
        lower bounds
    np.ndarray
        upper bounds
    """
    dim = A.shape[1]

    model = pmo.block()

    variables = []
    for i in range(dim):
        setattr(model, f"x{i}", pmo.variable())
        variables.append(getattr(model, f"x{i}"))

    model.A = pmo.matrix_constraint(A, ub=b, x=variables, sparse=True)

    opt = pmo.SolverFactory(solver_name)

    lower_upper_bounds = []
    for sense in [pmo.minimize, pmo.maximize]:

        bounds = []
        for i in range(dim):
            model.objective = pmo.objective(getattr(model, f"x{i}"), sense=sense)
            result = opt.solve(model)
            assert str(result.solver.termination_condition) == "optimal"
            bounds.append(result["Problem"][0]["Lower bound"])
            del model.objective

        bounds = np.array(bounds).reshape(len(bounds), 1)
        lower_upper_bounds.append(bounds)

    return tuple(lower_upper_bounds)


def sample_bounding_box(lb, ub, N=1e5):
    """
    Generate random samples in bounding box.

    Parameters
    ----------
    lb : np.ndarray
    ub : np.ndarray

    Returns
    -------
    np.ndarray
    """

    if not isinstance(N, int):
        N = int(N)

    dim = lb.shape[0]

    # TODO: use low-discrepancy series https://chaospy.readthedocs.io/en/master/sampling/sequences.html
    return np.tile(lb, (1, N)) + create_halton_samples(N, dim) * np.tile(
        ub - lb, (1, N)
    )


def residuals(A, b, samples):
    """
    Calculate residuals for all samples s
    r[s] = A*samples[s] - b

    Parameters
    ----------
    A : np.ndarray
    b : np.ndarray
    samples : np.ndarray

    Returns
    -------
    np.ndarray
        Residuals, dimensions constraints * samples
    """

    N = samples.shape[1]

    return np.dot(A, samples) - np.tile(np.array([b]).T, (1, N))


def volume(A, b, N=1e5):
    """
    Find volume of a polytope.

    Parameters
    ----------
    A : np.ndarray
        e.g. lhs of polytope for N-X secured network
    b : np.ndarray
        e.g. rhs of polytope for N-X secured network
    N: int|float, default 1e5
        number of random samples, e.g 1e5

    Returns
    -------
    vol: float
    """

    lb, ub = bounding_box(A, b)
    samples = sample_bounding_box(lb, ub, N)
    r = residuals(A, b, samples)
    z = np.all(r < 0, 0).sum()
    vol = np.prod(ub - lb) * z / N

    return vol


def polytope_n0(n, i):
    """
    Find polytope of N-0 secured network.

    Parameters
    ----------
    n : pypsa.Network
    i: int
       number of subnworks (e.g. 0, 1, 2, etc.)

    Returns
    -------
    A : np.ndarray
    b : np.ndarray
    """

    i = int(i)

    calculate_all_PTDF(n)

    sub = n.sub_networks.obj[i]

    PTDFr = sub.PTDF[:, 1:]
    s_nom = sub.branches().s_nom.values

    A = np.vstack((PTDFr, -PTDFr))
    b = np.hstack((s_nom, s_nom))

    return A, b


def apply_outage(n, line):
    """Removes outaged line by
    (a) removing line completely or
    (b) reducing num_parallel.

    Parameters
    ----------
    n : pypsa.Network
    line : str

    Returns
    -------
    no : pypsa.Network
    """

    nump = n.lines.at[line, "num_parallel"]
    if nump > 1:
        new_nump = nump - 1
    elif nump == 1:
        new_nump = 0
    elif nump > 0.5:
        new_nump = nump - 3 ** (-1)
    elif nump <= 0.5:
        new_nump = 0

    no = n.copy(with_time=False)

    if new_nump == 0:
        no.remove("Line", line)
    else:
        no.lines.at[line, "num_parallel"] = new_nump

    return no


def polytope_n1(n, i):
    """
    Find polytope of N-1 secured network.

    Parameters
    ----------
    n : pypsa.Network
    i: int
       number of subnworks (e.g. 0, 1, 2, etc.)

    Returns
    -------
    A : np.ndarray
    b : np.ndarray
    """

    A_list = []
    b_list = []

    sub = n.sub_networks.obj[i]
    i_b = int(0)

    for line in n.lines.index:
        if (n.lines.at[line, "bus0"] == sub.branches().bus0[i_b]) & (
            n.lines.at[line, "bus1"] == sub.branches().bus1[i_b]
        ):
            i_b += 1

            no = apply_outage(n, line)

            Ao, bo = polytope_n0(no, i)

            A_list.append(Ao)

            b_list.append(bo)

            del no
            if i_b == len(sub.branches()):
                A = np.vstack(A_list)
                b = np.hstack(b_list)
                break

    return A, b


def volumes(A1, b1, A2, b2, N=1e5):
    """
    Find volumes of various overlaps of
    polytope 1 (N-0 secured network with buffer) and
    polytope 2 (N-1 secured network).

    Parameters
    ----------
    A1 : np.ndarray
        lhs of polytope for N-0 secured network (with buffer)
    b1 : np.ndarray
        rhs of polytope for N-0 secured network (with buffer)
    A2 : np.ndarray
        lhs of polytope for N-1 secured network
    b2 : np.ndarray
        rhs of polytope for N-1 secured network
    N : int|float, default 1e5
        number of random samples

    Returns
    -------
    vol_1 : float
        volume of polytope 1
    vol_2 : float
        volume of polytope 2
    vol_1_in_2 : float
        volume of polytope 1 inside polytope 2
    vol_1_in_2_norm_1 : float
        vol_1_in_2 / vol_1
    vol_1_in_2_norm_2 : float
        vol_1_in_2 / vol_2
    vol_2_notin_1 : float
        volume for the part of polytope 2 which is outside of polytope 1
    vol_1_notin_2 : float
        volume for the part of polytope 1 which is outside of polytope 2
    z_bbox2_in_1 : int
        number of samples of bounding box of polytope 2 in polytope 1
    z_bbox2_in_2 : int
        number of samples of bounding box of polytope 2 in polytope 2
    z_bbox2_in_1_and_2 : int
        number of samples of bounding box of polytope 2 in polytope 1 and polytope 2
    """

    # TODO: reduce repeated recalculation of N-1 polytope in `find_c()`.
    # TODO: normalisation of volume to avoid too large values?

    if not isinstance(N, int):
        N = int(N)

    lb1, ub1 = bounding_box(A1, b1)
    bbox1 = sample_bounding_box(lb1, ub1, N)
    res1 = residuals(A1, b1, bbox1)
    bbox1_in_1 = np.all(res1 < 0, 0)
    z_bbox1_in_1 = bbox1_in_1.sum()

    lb2, ub2 = bounding_box(A2, b2)
    bbox2 = sample_bounding_box(lb2, ub2, N)
    res2 = residuals(A2, b2, bbox2)
    bbox2_in_2 = np.all(res2 < 0, 0)
    z_bbox2_in_2 = bbox2_in_2.sum()

    res2_bbox1 = residuals(A2, b2, bbox1)
    bbox1_in_2 = np.all(res2_bbox1 < 0, 0)
    z_bbox1_in_2 = bbox1_in_2.sum()

    res1_bbox2 = residuals(A1, b1, bbox2)
    bbox2_in_1 = np.all(res1_bbox2 < 0, 0)
    z_bbox2_in_1 = bbox2_in_1.sum()

    z_bbox1_in_1_and_2 = (bbox1_in_1 & bbox1_in_2).sum()
    z_bbox2_in_1_and_2 = (bbox2_in_1 & bbox2_in_2).sum()

    z_bbox1_in_1_and_not_2 = bbox1_in_1.sum() - z_bbox1_in_1_and_2

    def _vol(ub, lb, z, N):
        return np.prod(ub - lb) * z / N

    vol_1 = _vol(ub1, lb1, z_bbox1_in_1, N)
    vol_2 = _vol(ub2, lb2, z_bbox2_in_2, N)
    vol_1_in_2 = _vol(ub1, lb1, z_bbox1_in_1_and_2, N)
    vol_1_notin_2 = _vol(ub1, lb1, z_bbox1_in_1_and_not_2, N)
    vol_2_notin_1 = vol_2 - vol_1_in_2

    vol_1_in_2_norm_2 = vol_1_in_2 / vol_2
    vol_1_in_2_norm_1 = vol_1_in_2 / vol_1

    # TODO: potentially export as dictionary
    return (
        vol_1,
        vol_2,
        vol_1_in_2,
        vol_1_in_2_norm_1,
        vol_1_in_2_norm_2,
        vol_1_notin_2,
        vol_2_notin_1,
        z_bbox2_in_1,
        z_bbox2_in_2,
        z_bbox2_in_1_and_2,
    )


def extract_country(n, i, A, ct):
    """
    Reduce dimension of polytope to buses
    within a specified country.

    Parameters
    ----------
    sub_network : pypsa.subnetwork
        (e.g. n.sub_networks.obj[0], ..., n.sub_networks.obj[i])
    i: int
       number of subnworks (e.g. 0, 1, 2, etc.)
    A : np.ndarray
        e.g. lhs of N-X network polytope
    ct : str
        2-digit country code, e.g. 'DE'


    Returns
    -------
    A_cnt : np.ndarray
            reduced lhs of N-X network polytope

    """
    sub = n.sub_networks.obj[i]

    x_ct = []
    y_ct = []
    in_ct = []
    ctr = 0
    for j, (b_j, b) in enumerate(sub.buses()[1:].iterrows()):
        if b.country == ct:
            ctr += 1
            in_ct.append(j)
            x_ct.append(sub.buses().x[j + 1])
            y_ct.append(sub.buses().y[j + 1])

    return A[:, in_ct], in_ct, x_ct, y_ct, ctr


def break_condition_approximate(
    vout,
    min_ratio=0.99999,
    max_ratio=1.000001,
    min_overlap=0.9999999,
    max_overlap=1.000000001,
):
    """
    Evaluate break condition for approximate approach.

    Parameters
    ----------
    vout : tuple
        output of volumes()
    min_ratio : float, default 0.9
    max_ratio : float, default 1.1
    min_overlap : float, default 0.9
    max_overlap : float, default 1.1

    Returns
    -------
    bool
    """

    vol_1 = vout[0]
    vol_2 = vout[1]

    vol_1_notin_2 = vout[5]
    vol_2_notin_1 = vout[6]

    ratio = vol_1 / vol_2
    overlap = vol_1_notin_2 / vol_2_notin_1

    print("ration=", ratio)  # updated
    print("overlap=", overlap)  # updated

    return (
        (ratio <= max_ratio)
        & (ratio >= min_ratio)
        & (overlap <= max_overlap)
        & (overlap >= min_overlap)
    ) or (  # updated
        (ratio <= max_ratio)  # updated
        & (ratio >= min_ratio)  # updated
        & (overlap == 0)
    )  # updated


def break_condition_robust(vout):
    """
    Evaluate break condition for robust approach.

    Parameters
    ----------
    vout : tuple
        output of volumes()

    Returns
    -------
    bool
    """

    z_bbox2_in_1 = vout[7]
    z_bbox2_in_1_and_2 = vout[9]

    return z_bbox2_in_1 != z_bbox2_in_1_and_2


def selection_sort(xx):
    for i in range(len(xx)):
        swap = i + np.argmin(xx[i:])
        (xx[i], xx[swap]) = (xx[swap], xx[i])
    return xx


def cluster_country(n, i, A, ct, cluster):

    A_ct, b_ct, x_ct, y_ct, ctr = extract_country(n, i, A, ct)

    cluster_no = len(n.lines)

    print("for line", n.lines.index[cluster])

    i_bus0 = None
    i_bus1 = None

    bus0 = n.lines.bus0.loc[n.lines.index[cluster]]
    bus1 = n.lines.bus1.loc[n.lines.index[cluster]]

    for ib in range(1, len(n.buses)):
        if n.buses.index[ib] == bus0:
            i_bus0 = ib - 1
        elif n.buses.index[ib] == bus1:
            i_bus1 = ib - 1

    if (i_bus0 != None) & (i_bus1 != None):
        i_bus = [i_bus0, i_bus1]
    elif (i_bus0 == None) & (i_bus1 != None):
        i_bus = [i_bus1]
    elif (i_bus0 != None) & (i_bus1 == None):
        i_bus = [i_bus0]

    A_cluster = A_ct[:, i_bus]
    return A_cluster, cluster_no, i_bus


def plot_cluster(n, i, A, ct):

    A_ct, b_ct, x_ct, y_ct, ctr = extract_country(n, i, A, ct)

    s = (ctr, 2)
    X = np.zeros(s)

    for k in range(0, ctr):
        X[k, 0] = x_ct[k]
        X[k, 1] = y_ct[k]

    bic = int(4)
    if ctr % bic != 0:
        n_cluster = int(ctr / bic) + 1
    else:
        n_cluster = int(ctr / bic)

    km = KMeans(
        n_clusters=n_cluster,
        init="random",  # desired cluster=3
        n_init=10,
        max_iter=300,  # max_iter=300 maximum number of iterations
        tol=1e-04,
        random_state=0,  # tol = tolerance 0.0001
    )
    y_km = km.fit_predict(X)
    # plot clusters
    for z in range(0, max(y_km) + 1):
        plt.scatter(
            X[y_km == z, 0],
            X[y_km == z, 1],
            s=50,
            marker="s",
            edgecolor="black",  # ,label=z
        )

    # plot the centroids
    plt.scatter(
        km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 1],
        s=250,
        marker="*",
        c="red",
        edgecolor="black",
        # label='centroids'
    )
    # plt.legend(scatterpoints=1)
    plt.grid()
    plt.savefig("fig1.png")
    # plt.show()


def contingency_factor_cluster(n, i, ct, c_start=0.0, c_end=1.0, app="robust", N=1e5):
    A_n0, b_n0 = polytope_n0(n, i)
    A_n0_ct, cluster_size, randomlist = cluster_country(n, i, A_n0, ct, cluster=0)

    #     cluster_list=[]
    c = []
    for clster in range(0, cluster_size):
        c_cluster, vol_cluster, vol_prog_cluster = contingency_factor(
            n,
            0,
            "DE",
            approach=app,
            c_start=c_start,
            c_end=c_end,
            N=int(N),
            cluster=clster,
        )
        c.append(c_cluster)
        #         cluster_list.append(line_index)
        print(clster, c_cluster)

    return c, cluster_list


def contingency_factor(
    n,
    i,
    ct=None,
    approach="approximate",
    c_start=0.0,
    c_end=1.0,
    c_step=0.01,
    N=1e5,
    A_n0=None,
    b_n0=None,
    A_n1=None,
    b_n1=None,
    cluster=None,
    break_kwargs={},
):
    """
    Find contingency factors based on
    approximate or robust for the whole
    network or a particular country.

    Parameters
    ----------
    n : pypsa.Network
    i: int
       number of subnworks (e.g. 0, 1, 2, etc.)
    ct : str, default None
        Name of the country, e.g. "DE".
    approach : str
        Approach for break condition.
        Can be "approximate" or "robust".
    c_start : float, default 0.0
        Initial contingency factor, [0.,0.99]
    c_end : float, default 1.0
        Final contingency factor, [0.01,1.0]
    c_step : float, default 0.01
        Step size for contingency factor loop, [0.01,1.0]
    N : int|float, default 1e5
        Number of random samples
    A_n0 : np.ndarray, default None
    b_n0 : np.ndarray, default None
    A_n1 : np.ndarray, default None
    b_n1 : np.ndarray, default None
    cluster: , default None
    break_kwargs : dict, default {}
        Additional arguments for evaluating break condition.
        For approximate, these are min_ratio, max_ratio,
        min_overlap, max_overlap. For example:
        {
            "min_ratio": 0.9,
            "max_ratio": 1.1,
            "min_overlap": 0.9,
            "max_overlap": 1.1
        }

    Returns
    -------
    c : float
        contingency factor for selected country/ies
    vol_1_in_2_norm_2 : float
    vol_progress : dict
        dictionary of progression of vol_1_in_2_norm_2
        from c_start to c_end
    """

    assert approach in [
        "approximate",
        "robust",
    ], f"Approach must be one of ['robust', 'approximate'] but is {approach}."
    break_condition = globals()[f"break_condition_{approach}"]

    if not isinstance(N, int):
        N = int(N)

    if A_n0 is None or b_n0 is None:
        A_n0, b_n0 = polytope_n0(n, i)

    if A_n1 is None or b_n1 is None:
        A_n1, b_n1 = polytope_n1(n, i)

    if ct is not None:
        if cluster is not None:
            A_n0_ct, cluster_size, randomlist = cluster_country(n, i, A_n0, ct, cluster)
            A_n1_ct_ign, b_ct_ign, x_ct_ign, y_ct_ign, ctr_ign = extract_country(
                n, i, A_n1, ct
            )
            A_n1_ct = A_n1_ct_ign[:, randomlist]
        else:
            A_n0_ct, b_n0_ct, x_n0_ct, y_n0_ct, ctr_n0 = extract_country(n, i, A_n0, ct)
            A_n1_ct, b_n1_ct, x_n1_ct, y_n1_ct, ctr_n1 = extract_country(n, i, A_n1, ct)
    else:
        A_n0_ct, A_n1_ct = A_n0, A_n1

    vol_progress = {}
    for c in np.arange(c_start, c_end, c_step):

        c = np.round(c, decimals=3)
        logger.info(f"Check contingency factor {c} for {approach}.")
        b_n0_c = c * b_n0

        vout = volumes(A_n0_ct, b_n0_c, A_n1_ct, b_n1, N)
        vol_progress[c] = vout[4]

        if break_condition(vout, **break_kwargs):
            logger.info(f"Break condition met for {approach} at {c}")
            if approach == "robust":
                c -= c_step
            break
    return (c, vout[4], vol_progress)


def calculate_gl_line(n, i):
    """
    Find redundant capacity for each line in each subnetwork.

    Parameters
    ----------
    n : pypsa.Network
    i : int,
        Number of the subnetwork, e.g. 0, 1, etc.

    Returns
    -------
    gl_line : pandas.core.frame.DataFrame
        redundant capacity for each line
    S_new : pandas.core.frame.DataFrame
        Modified S which is get: S(line) = gl_line(line)*S_nom(line)
    """

    i = int(i)
    sub = n.sub_networks.obj[i]
    i_b = int(0)

    gl_line = pd.DataFrame(
        {"gl_of_line": np.zeros(len(sub.branches()))}, index=sub.branches().index
    )

    S_new = pd.DataFrame(
        {"S_modified": np.zeros(len(sub.branches()))}, index=sub.branches().index
    )

    for line in n.lines.index:
        if (n.lines.at[line, "bus0"] == sub.branches().bus0[i_b]) & (
            n.lines.at[line, "bus1"] == sub.branches().bus1[i_b]
        ):
            # select the index/name of the line you want to remove
            bus0_removed_line = n.lines.bus0.loc[line]
            bus1_removed_line = n.lines.bus1.loc[line]
            P_add_max = n.lines.s_nom.loc[line]

            no = apply_outage(n, line)

            loop_int = 0

            loop_end = int(1e6)

            for i_ct in range(loop_int, loop_end):
                g_l = 0.01 * i_ct
                no.add(
                    "Generator",
                    "gen_add_{}".format(line),
                    bus=bus0_removed_line,
                    p_set=g_l * P_add_max,
                    control="PQ",
                )

                no.add(
                    "Load",
                    "load_add_{}".format(line),
                    bus=bus1_removed_line,
                    p_set=g_l * P_add_max,
                )

                ##################

                no.lpf()

                ##################

                p_line_norm = abs((no.lines_t.p0) / (no.lines.s_nom))
                if np.all(p_line_norm.loc["now"] <= 1.0) != 1:
                    gl_opt = g_l - 0.01
                    gl_line.gl_of_line[sub.branches().index[i_b]] = gl_opt
                    S_new.S_modified[sub.branches().index[i_b]] = (
                        gl_opt * n.lines["s_nom"][line]
                    )
                    break

                load_to_remove = "load_add_{}".format(line)
                gen_to_remove = "gen_add_{}".format(line)
                no.remove("Load", load_to_remove)
                no.remove("Generator", gen_to_remove)

            ##################
            i_b += 1
            if i_b == len(sub.branches()):
                break

    print("gl_line and S_new found!")
    print("-----------------------------------")
    return gl_line, S_new


def find_c_robust_gl(
    n,
    i,
    ct=None,
    approach="approximate",
    c_start=0.0,
    c_end=1.0,
    c_step=0.01,
    N=1e5,
    A_n0=None,
    b_n0=None,
    A_n1=None,
    b_n1=None,
    break_kwargs={},
):
    """
    Find contingency factors based on
    approximate or robust for the whole
    network or a particular country.

    Parameters
    ----------
    n : pypsa.Network
    i : int,
        Number of the subnetwork, e.g. 0, 1, etc.
    ct : str, default None
        Name of the country, e.g. "DE".
    approach : str
        Approach for break condition.
        Can be "approximate" or "robust".
    c_start : float, default 0.0
        Initial contingency factor, [0.,0.99]
    c_end : float, default 1.0
        Final contingency factor, [0.01,1.0]
    c_step : float, default 0.01
        Step size for contingency factor loop, [0.01,1.0]
    N : int|float, default 1e5
        Number of random samples
    A_n0 : np.ndarray, default None
    b_n0 : np.ndarray, default None
    A_n1 : np.ndarray, default None
    b_n1 : np.ndarray, default None
    break_kwargs : dict, default {}
        Additional arguments for evaluating break condition.
        For approximate, these are min_ratio, max_ratio,
        min_overlap, max_overlap. For example:
        {
            "min_ratio": 0.9,
            "max_ratio": 1.1,
            "min_overlap": 0.9,
            "max_overlap": 1.1
        }

    Returns
    -------
    c : float
        contingency factor for selected country/ies
    vol_1_in_2_norm_2 : float
    vol_progress : dict
        dictionary of progression of vol_1_in_2_norm_2
        from c_start to c_end
    gl_line : pandas.core.frame.DataFrame
        redundant capacity for each line
    """
    assert approach in [
        "approximate",
        "robust",
    ], f"Approach must be one of ['robust', 'approximate'] but is {approach}."
    break_condition = globals()[f"break_condition_{approach}"]

    if not isinstance(N, int):
        N = int(N)

    if A_n0 is None or b_n0 is None:
        A_n0, b_n0 = polytope_n0(n, i)

    if A_n1 is None or b_n1 is None:
        A_n1, b_n1 = polytope_n1(n, i)

    if ct is not None:
        A_n0_ct = extract_country(n, i, A_n0, ct)
        A_n1_ct = extract_country(n, i, A_n1, ct)
    else:
        A_n0_ct, A_n1_ct = A_n0, A_n1

    n_c = n.copy(with_time=False)

    sub_c = n_c.sub_networks.obj[i]

    i_b = int(0)

    gl_line, S_new = calculate_gl_line(n_c, i)

    for line in n_c.lines.index:
        if (n_c.lines.at[line, "bus0"] == sub_c.branches().bus0[i_b]) & (
            n_c.lines.at[line, "bus1"] == sub_c.branches().bus1[i_b]
        ):
            n_c.lines["s_nom"][line] = S_new.S_modified[sub_c.branches().index[i_b]]
            i_b += 1
            if i_b == len(sub_c.branches()):
                break

    b_c, s_nom = calculate_s_nom(n_c, i)

    vol_progress = {}

    for c in np.arange(c_start, c_end, c_step):

        c = np.round(c, decimals=3)

        logger.info(f"Check contingency factor {c} for {approach}.")

        b_n0_c = c * b_c

        vout = volumes(A_n0_ct, b_n0_c, A_n1_ct, b_n1, N)

        vol_progress[c] = vout[4]

        if break_condition(vout, **break_kwargs):
            logger.info(f"Break condition met for {approach} at {c}")
            if approach == "robust":
                c -= c_step
            break

    print("c_robust_gl found!")
    print("-----------------------------------")
    return (c, vout[4], vol_progress, gl_line)


def calculate_initial_c_line(
    n,
    i,
    ct,
    c_start=0.0,
    c_end=1.0,
    N=1e5,
    A_n0=None,
    b_n0=None,
    A_n1=None,
    b_n1=None,
    break_kwargs={},
):

    """
    Find contingency factors based on
    approximate or robust for the whole
    network or a particular country.

    Parameters
    ----------
    n : pypsa.Network
    i : int,
        Number of the subnetwork, e.g. 0, 1, etc.
    ct : str, default None
        Name of the country, e.g. "DE".
    approach : str
        Approach for break condition.
        Can be "approximate" or "robust".
    c_start : float, default 0.0
        Initial contingency factor, [0.,0.99]
    c_end : float, default 1.0
        Final contingency factor, [0.01,1.0]
    c_step : float, default 0.01
        Step size for contingency factor loop, [0.01,1.0]
    N : int|float, default 1e5
        Number of random samples
    A_n0 : np.ndarray, default None
    b_n0 : np.ndarray, default None
    A_n1 : np.ndarray, default None
    b_n1 : np.ndarray, default None
    break_kwargs : dict, default {}
        Additional arguments for evaluating break condition.
        For approximate, these are min_ratio, max_ratio,
        min_overlap, max_overlap. For example:
        {
            "min_ratio": 0.9,
            "max_ratio": 1.1,
            "min_overlap": 0.9,
            "max_overlap": 1.1
        }

    Returns
    -------
    c_rob_uni : float
        robust contingency factor for selected country/ies
    C_r_line : pandas.core.frame.DataFrame
        initial contingency factor for each line

    """

    c_rob, v, vol, gl_line = find_c_robust_gl(
        n, i, ct, approach="robust", c_start=c_start, c_end=c_end, N=N
    )

    c_rob_uni, v, vol = contingency_factor(
        n, i, ct, approach="robust", c_start=c_start, c_end=c_end, N=N
    )

    sub = n.sub_networks.obj[i]

    i_b = int(0)

    C_r_line = pd.DataFrame(
        {"C_of_line": np.zeros(len(sub.branches()))}, index=sub.branches().index
    )

    for line in n.lines.index:
        if (n.lines.at[line, "bus0"] == sub.branches().bus0[i_b]) & (
            n.lines.at[line, "bus1"] == sub.branches().bus1[i_b]
        ):
            C_r_line.C_of_line[sub.branches().index[i_b]] = (
                c_rob * gl_line.gl_of_line[sub.branches().index[i_b]]
            )

            if C_r_line.C_of_line[sub.branches().index[i_b]] < c_rob_uni:
                C_r_line.C_of_line[sub.branches().index[i_b]] = c_rob_uni

            elif C_r_line.C_of_line[sub.branches().index[i_b]] > 1:
                C_r_line.C_of_line[sub.branches().index[i_b]] = 1

            i_b += 1
            if i_b == len(sub.branches()):
                break

    return (C_r_line, c_rob_uni)


def calculate_c_line(
    n,
    i,
    ct=None,
    approach="approximate",
    c_start=0.0,
    c_end=1.0,
    c_step=0.01,
    N=1e5,
    A_n0=None,
    b_n0=None,
    A_n1=None,
    b_n1=None,
    break_kwargs={},
):
    """
    Find contingency factors based on
    approximate or robust for the whole
    network or a particular country.

    Parameters
    ----------
    n : pypsa.Network
    i : int,
        Number of the subnetwork, e.g. 0, 1, etc.
    ct : str, default None
        Name of the country, e.g. "DE".
    approach : str
        Approach for break condition.
        Can be "approximate" or "robust".
    c_start : float, default 0.0
        Initial contingency factor, [0.,0.99]
    c_end : float, default 1.0
        Final contingency factor, [0.01,1.0]
    c_step : float, default 0.01
        Step size for contingency factor loop, [0.01,1.0]
    N : int|float, default 1e5
        Number of random samples
    A_n0 : np.ndarray, default None
    b_n0 : np.ndarray, default None
    A_n1 : np.ndarray, default None
    b_n1 : np.ndarray, default None
    break_kwargs : dict, default {}
        Additional arguments for evaluating break condition.
        For approximate, these are min_ratio, max_ratio,
        min_overlap, max_overlap. For example:
        {
            "min_ratio": 0.9,
            "max_ratio": 1.1,
            "min_overlap": 0.9,
            "max_overlap": 1.1
        }

    Returns
    -------
    c_line : pandas.core.frame.DataFrame
             contingency factor for each line
    """

    assert approach in [
        "approximate",
        "robust",
    ], f"Approach must be one of ['robust', 'approximate'] but is {approach}."
    break_condition = globals()[f"break_condition_{approach}"]

    if not isinstance(N, int):
        N = int(N)

    if A_n0 is None or b_n0 is None:
        A_n0, b_n0 = polytope_n0(n, i)

    if A_n1 is None or b_n1 is None:
        A_n1, b_n1 = polytope_n1(n, i)

    if ct is not None:
        A_n0_ct = extract_country(n, i, A_n0, ct)
        A_n1_ct = extract_country(n, i, A_n1, ct)
    else:
        A_n0_ct, A_n1_ct = A_n0, A_n1

    vol_progress = {}

    C_r_line, c_rob_uni = calculate_initial_c_line(
        n, i, ct, c_start=c_start, c_end=c_end, N=N
    )

    c_line = C_r_line

    loop_int = 1

    loop_end = int(1 + ((1 - c_rob_uni) * 100))

    leng_loop = loop_end - loop_int

    print("c_rob_uni is:")
    print(c_rob_uni)
    print("C_r_line is:")
    print(C_r_line)
    print("-------")
    print("ready")
    print("-------")
    done = 0
    sub = n.sub_networks.obj[i]
    for i_ct in range(loop_int, loop_end):

        if done == 1:
            print("Done!")
            print(c_line)
            break

        i_b = int(0)
        for line in n.lines.index:
            if (n.lines.at[line, "bus0"] == sub.branches().bus0[i_b]) & (
                n.lines.at[line, "bus1"] == sub.branches().bus1[i_b]
            ):
                if C_r_line.C_of_line[sub.branches().index[i_b]] > c_rob_uni:
                    C_r_line.C_of_line[sub.branches().index[i_b]] -= 0.01
                else:
                    C_r_line.C_of_line[sub.branches().index[i_b]] = C_r_line.C_of_line[
                        sub.branches().index[i_b]
                    ]

                c_line = C_r_line
                i_ct = np.round(i_ct, decimals=3)
                print(i_ct, "-----Next----", line)
                print(c_line)
                print("-----------")
                logger.info(f"Check contingency factor {i_ct} for {approach}.")
                n_c = n.copy()
                b_c, s_nom = calculate_s_nom(n_c, i)
                s_nom_g_l = np.multiply(np.hstack(C_r_line.C_of_line), s_nom)
                b_n0_c = np.hstack((s_nom_g_l, s_nom_g_l))
                vout = volumes(A_n0_ct, b_n0_c, A_n1_ct, b_n1, N)
                vol_progress[i_ct] = vout[4]
                z_bbox2_in_1 = vout[7]
                z_bbox2_in_1_and_2 = vout[9]

                #                 if z_bbox2_in_1 != z_bbox2_in_1_and_2:
                #                     print('Out')
                #                 else:
                #                     print('IN')

                #                 print('-----------')

                if z_bbox2_in_1 == z_bbox2_in_1_and_2:
                    done = 1
                    break
                else:
                    i_b += 1
                    if i_b == len(sub.branches()):
                        break

    #             if done == 1:
    #                 print('Done!')
    #                 print(c_line)
    #                 break
    #     if done == 1:
    #         break

    return c_line
