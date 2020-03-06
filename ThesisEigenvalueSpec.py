#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.createStimulus import *
from modules.networkAnalysis import *
from createThesisNetwork import create_network, NETWORK_TYPE
from modules.thesisUtils import arg_parse

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

    (_,
     _,
     adj_sens_sens_mat,
     _,
     _,
     _) = create_network(
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
        fig_name="%s_network_non-zero_connections.png" % network_type
    )


if __name__ == '__main__':
    cmd_params = arg_parse()
    if cmd_params.seed:
        np.random.seed(0)
    if cmd_params.agg:
        import matplotlib
        matplotlib.use("Agg")

    for network_type in list(NETWORK_TYPE.keys()):
        main_eigenvalue_spec(network_type=network_type)


