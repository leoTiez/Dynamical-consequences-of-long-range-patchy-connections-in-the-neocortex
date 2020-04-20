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
        num_neurons=int(1e4),
        save_plot=False,
        verbosity=VERBOSITY
):
    """
    The main function to compute eigenvalue spectrum
    :param network_type: The network type. This must be an integer number defined in the NETWORK_TYPE dictionary
    :param num_neurons: Number of sensory neurons
    :param save_plot: If set to True the plot is saved. If False the plot is displayed
    :param verbosity: Verbosity flag
    :return: None
    """
    # load input stimulus
    stimulus_size = (50, 50)
    input_stimulus = stimulus_factory(input_type=INPUT_TYPE["plain"], size=stimulus_size)
    if verbosity > 2:
        plt.imshow(input_stimulus, cmap='gray')
        plt.show()

    # #################################################################################################################
    # Define values
    # #################################################################################################################
    num_neurons = num_neurons

    network = network_factory(
        input_stimulus,
        network_type=network_type,
        num_sensory=num_neurons,
        verbosity=verbosity
    )
    network.create_network()
    sens_weight_mat = network.get_sensory_weight_mat()

    _, _ = eigenvalue_analysis(
        sens_weight_mat,
        plot=True,
        save_plot=save_plot,
        fig_name="%s_network_connections.png" % NETWORK_TYPE.keys()[network_type]
    )


def main():
    # #################################################################################################################
    # Initialise parameters
    # #################################################################################################################
    verbosity = VERBOSITY
    num_neurons = int(1e4)
    save_plot=True
    networks = NETWORK_TYPE.keys()

    # #################################################################################################################
    # Parse command line arguments
    # #################################################################################################################
    cmd_params = arg_parse()
    if cmd_params.seed:
        np.random.seed(0)

    if cmd_params.agg:
        import matplotlib
        matplotlib.use("Agg")

    if cmd_params.show:
        save_plot=False

    if cmd_params.network is not None:
        if cmd_params.network in list(NETWORK_TYPE.keys()):
            networks = [cmd_params.network]
        else:
            raise ValueError("Please pass a valid network as parameter")

    if cmd_params.verbosity is not None:
        verbosity = cmd_params.verbosity

    if cmd_params.num_neurons is not None:
        num_neurons = cmd_params.num_neurons

    # #################################################################################################################
    # Run the eigenvalue spectra analysis
    # #################################################################################################################
    for network_type in networks:
        main_eigenvalue_spec(
            network_type=NETWORK_TYPE[network_type],
            num_neurons=num_neurons,
            save_plot=save_plot,
            verbosity=verbosity
        )


if __name__ == '__main__':
    main()


