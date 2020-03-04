#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.createStimulus import *
from modules.networkAnalysis import *
from createThesisNetwork import create_network, NETWORK_TYPE

import matplotlib.pyplot as plt

import nest


VERBOSITY = 1
nest.set_verbosity("M_ERROR")


def main_eigenvalue_spec(network_type, shuffle_input=False):
    # load input stimulus
    input_stimulus = image_with_spatial_correlation(
        size_img=(50, 50),
        num_circles=5,
        radius=10,
        background_noise=shuffle_input,
        shuffle=shuffle_input
    )

    # input_stimulus = create_image_bar(0, shuffle=shuffle_input)
    # input_stimulus = load_image("nfl-sunflower50.jpg")
    if VERBOSITY > 2:
        plt.imshow(input_stimulus, cmap='gray')
        plt.show()

    # #################################################################################################################
    # Define values
    # #################################################################################################################
    plot_eigenvalue_spec = False
    cap_s = 1.     # Increased to reduce the effect of the input and to make it easier to investigate the dynamical
                    # consequences of local / lr patchy connections

    (torus_layer,
     adj_rec_sens_mat,
     adj_sens_sens_mat,
     tuning_weight_vector,
     spike_detect) = create_network(
        input_stimulus,
        sens_adj_mat_needed=True,
        cap_s=cap_s,
        network_type=network_type,
        verbosity=VERBOSITY
    )

    _, _ = eigenvalue_analysis(
        adj_sens_sens_mat,
        plot=True,
        save_plot=plot_eigenvalue_spec,
        fig_name="thesis_network_non-zero_connections.png"
    )


if __name__ == '__main__':
    main_eigenvalue_spec(network_type="local_radial_lr_patchy")


