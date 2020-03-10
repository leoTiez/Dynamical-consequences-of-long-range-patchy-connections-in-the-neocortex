#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.createStimulus import *
from modules.networkConstruction import *
from modules.thesisUtils import arg_parse
from createThesisNetwork import create_network

import numpy as np
import matplotlib.pyplot as plt

import nest


VERBOSITY = 2
nest.set_verbosity("M_ERROR")


def main_matrix_dynamics(network_type="local_radial_lr_patchy"):
    nest.ResetKernel()
    input_stimulus = image_with_spatial_correlation(size_img=(50, 50), num_circles=5, background_noise=False)
    if VERBOSITY > 2:
        plt.imshow(input_stimulus, cmap='gray')
        plt.show()
    # #################################################################################################################
    # Define values
    # #################################################################################################################
    cap_s = 1.
    plot_arrangement_rows = 5
    plot_arrangement_columns = 5

    (_, adj_rec_sens_mat, adj_sens_sens_mat, _, _, _, _) = create_network(
        input_stimulus,
        sens_adj_mat_needed=True,
        cap_s=cap_s,
        network_type=network_type,
        verbosity=VERBOSITY
    )

    # Sufficient to use only 255 as we don't use the neurons themselves
    input_matrix = np.full((50, 50), 255)
    sensory_activity = adj_rec_sens_mat.T.dot(input_matrix.reshape(-1))
    fig, axes = plt.subplots(plot_arrangement_rows, plot_arrangement_columns)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    for ax in axes.reshape(-1):
        ax.imshow(sensory_activity.reshape((50, 50)))
        sensory_activity = adj_sens_sens_mat.T.dot(sensory_activity)

    plt.show()


if __name__ == '__main__':
    cmd_params = arg_parse()
    if cmd_params.seed:
        np.random.seed(0)
    if cmd_params.agg:
        import matplotlib
        matplotlib.use("Agg")

    main_matrix_dynamics(network_type="local_radial_lr_patchy")


