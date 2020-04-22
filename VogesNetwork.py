#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ####################################################################################################################
# This Python script simulates the networks with long-range connections that are established according to different
# parameters, as described in the paper by Voges et al.:
#
# Voges, N., Guijarro, C., Aertsen, A. & Rotter, S.
# Models of cortical networks with long-range patchy projections.
# Journal of Computational Neuroscience 28, 137–154 (2010).
# DOI: 10.1007/s10827-009-0193-z
# ####################################################################################################################

# Own modules
from modules.networkConstruction import *
from modules.networkAnalysis import *

# External imports
import numpy as np
import matplotlib.pyplot as plt


# Nest libraries
import nest.topology as tp
import nest

# Define global constants
GLOBAL_CONNECTIVITY = 0.0123
R_MAX = 8.

NETWORK_DICT = {
    "np": 1,
    "random": 2,
    "overlapping": 3,
    "shared": 4,
    "partially-overlapping": 5
}


def create_distant_connections(torus_layer, connection_type=NETWORK_DICT["np"]):
    """
    Create network specific long-distance connections
    :param torus_layer: The neural sheet
    :param connection_type: The connection type. This is an integer number defined in the NETWORK_DICT dictionary
    :return: Nodes of the layer or the sublayer for debugging purposes
    """
    if connection_type == NETWORK_DICT["np"]:
        debug_layer = create_distant_np_connections(torus_layer)
    elif connection_type == NETWORK_DICT["random"]:
        debug_layer = create_random_patches(torus_layer)
    elif connection_type == NETWORK_DICT["overlapping"]:
        debug_layer = create_overlapping_patches(torus_layer)
    elif connection_type == NETWORK_DICT["shared"]:
        debug_layer = create_shared_patches(torus_layer)
    elif connection_type == NETWORK_DICT["partially-overlapping"]:
        debug_layer = create_partially_overlapping_patches(torus_layer)
    else:
        raise ValueError("Not a valid network")

    return debug_layer


def main_create_eigenspectra_plots():
    """
    Compute the eigenvalue spectra and plot them
    :return: None
    """
    torus_layer, _, _ = create_torus_layer_uniform()
    create_local_circular_connections_topology(torus_layer)

    for key in NETWORK_DICT:
        _ = create_distant_connections(torus_layer, connection_type=NETWORK_DICT[key])

        adj_mat = create_adjacency_matrix(nest.GetNodes(torus_layer)[0], nest.GetNodes(torus_layer)[0])
        eigenvalue_analysis(adj_mat, plot=True, save_plot=True, fig_name="voges_adj_matrix_%s.png" % key)


def main(
        plot_torus=True,
        plot_target=True,
        num_plot_tagets=3,
        use_lr_connection_type=NETWORK_DICT["np"]
):
    """
    Main function running the test routines
    :param plot_torus: Flag to plot neural layer
    :param plot_target: Flag to plot targets to control established connections
    :param num_plot_tagets: Plot connections of the num_plot_targts-th node
    :param use_lr_connection_type: Define the type of long range connections
    :return None
    """
    torus_layer = create_torus_layer_uniform()
    if plot_torus:
        fig, _ = plt.subplots()
        tp.PlotLayer(torus_layer, fig)
        plt.show()

    create_local_circular_connections(torus_layer)

    debug_layer = create_distant_connections(torus_layer, connection_type=use_lr_connection_type)

    if plot_target:
        choice = np.random.choice(np.asarray(debug_layer), num_plot_tagets, replace=False)
        for c in choice:
            tp.PlotTargets([int(c)], torus_layer)
        plt.show()

    adj_mat = create_adjacency_matrix(nest.GetNodes(torus_layer)[0], nest.GetNodes(torus_layer)[0])
    eigenvalue_analysis(adj_mat, plot=True)


if __name__ == '__main__':
    np.random.seed(0)
    # main(plot_torus=False, use_lr_connection_type=NETWORK_DICT['partially-overlapping'])
    main_create_eigenspectra_plots()

