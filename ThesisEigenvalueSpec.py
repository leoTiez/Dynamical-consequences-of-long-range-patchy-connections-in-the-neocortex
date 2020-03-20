#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.createStimulus import stimulus_factory, INPUT_TYPE
from modules.networkAnalysis import *
from createThesisNetwork import network_factory, NETWORK_TYPE
from modules.thesisUtils import arg_parse

import matplotlib.pyplot as plt

import nest


VERBOSITY = 1
nest.set_verbosity("M_ERROR")


def main_eigenvalue_spec(
        network_type=NETWORK_TYPE["local_circ_patchy_sd"],
        input_type=INPUT_TYPE["plain"],
        save_plot=False
):
    """
    The main function to compute eigenvalue spectrum
    :param network_type: The network type. This must be an integer number defined in the NETWORK_TYPE dictionary
    :param input_type: The input type. This must be an integer number defined in the INPUT_TYPE dictionary
    :param save_plot: If set to True the plot is saved. If False the plot is displayed
    :return: None
    """
    # load input stimulus
    stimulus_size = (50, 50)
    input_stimulus = stimulus_factory(input_type, size=stimulus_size)
    # input_stimulus = create_image_bar(0, shuffle=shuffle_input)
    # input_stimulus = load_image("nfl-sunflower50.jpg")
    if VERBOSITY > 2:
        plt.imshow(input_stimulus, cmap='gray')
        plt.show()

    # #################################################################################################################
    # Define values
    # #################################################################################################################
    num_neurons = int(1e4)

    network = network_factory(input_stimulus, network_type=network_type, num_sensory=num_neurons)
    network.create_network()
    sens_weight_mat = network.get_sensory_weight_mat()

    _, _ = eigenvalue_analysis(
        sens_weight_mat,
        plot=True,
        save_plot=save_plot,
        fig_name="%s_network_connections.png" % network_type
    )


def main():
    for network_type in list(NETWORK_TYPE.keys()):
        main_eigenvalue_spec(network_type=NETWORK_TYPE[network_type])


if __name__ == '__main__':
    cmd_params = arg_parse()
    if cmd_params.seed:
        np.random.seed(0)
    if cmd_params.agg:
        import matplotlib
        matplotlib.use("Agg")

    main()


