#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.stimulusReconstruction import fourier_trans, direct_stimulus_reconstruction
from modules.networkConstruction import *
from modules.createStimulus import *
from modules.networkAnalysis import *
from createThesisNetwork import create_network

import numpy as np
import matplotlib.pyplot as plt

import nest


VERBOSITY = 4
nest.set_verbosity("M_ERROR")


def main_matrix_dynamics():
    input_stimulus = image_with_spatial_correlation(size_img=(50, 50), num_circles=5, background_noise=False)
    if VERBOSITY > 2:
        plt.imshow(input_stimulus, cmap='gray')
        plt.show()
    # #################################################################################################################
    # Define values
    # #################################################################################################################
    cap_s = 1.
    receptor_connect_strength = 1.
    plot_arrangement_rows = 10
    plot_arrangement_columns = 10

    (_, _, adj_rec_sens_mat, adj_sens_sens_mat, _, _) = create_network(
        input_stimulus,
        cap_s=cap_s,
        receptor_connect_strength=receptor_connect_strength,
        ignore_weights_adj=False,
        use_stimulus_local=False,
        verbosity=VERBOSITY
    )

    # Sufficient to use only 255 as we don't use the neurons themselves
    input_matrix = np.full((50, 50), 255)
    sensory_activity = adj_rec_sens_mat.T.dot(input_matrix.reshape(-1))
    fig, axes = plt.subplots(plot_arrangement_rows, plot_arrangement_columns)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    for ax in axes.reshape(-1):
        ax.imshow(sensory_activity.reshape((10, 25)), cmap='gray')
        sensory_activity = adj_sens_sens_mat.T.dot(sensory_activity)

    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    main_matrix_dynamics()


