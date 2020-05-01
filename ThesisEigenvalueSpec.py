#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.createStimulus import stimulus_factory
from modules.networkAnalysis import *
from createThesisNetwork import network_factory
from modules.thesisUtils import arg_parse
from modules.thesisConstants import *

import sys
import matplotlib.pyplot as plt

import nest


VERBOSITY = 1
nest.set_verbosity("M_ERROR")


def main_eigenvalue_spec(
        network_type=NETWORK_TYPE["local_circ_patchy_sd"],
        num_neurons=int(1e4),
        patches=3,
        c_alpha=0.7,
        compute_sum=False,
        save_plot=False,
        verbosity=VERBOSITY
):
    """
    The main function to compute eigenvalue spectrum
    :param network_type: The network type. This must be an integer number defined in the NETWORK_TYPE dictionary
    :param num_neurons: Number of sensory neurons
    :param patches: Number of patches per neuron
    :param compute_sum: If set to true, the sum of the input weights is computed
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
        c_alpha=c_alpha,
        network_type=network_type,
        num_patches=patches,
        num_sensory=num_neurons,
        verbosity=verbosity
    )
    network.create_network()
    sens_weight_mat = network.get_sensory_weight_mat()

    if compute_sum:
        rec_sum = sens_weight_mat.sum(axis=0)
        ff_sum = network.ff_weight_mat.sum(axis=0)[:-1]
        print("\n#####################\t Mean input weight of recurrent weight matrix for %s: %s"
              % (list(NETWORK_TYPE.keys())[network_type], rec_sum.mean()))
        print("\n#####################\t Mean input weight of recurrent and ff weights %s: %s"
              % (list(NETWORK_TYPE.keys())[network_type], (rec_sum + ff_sum).mean()))

    _, _ = eigenvalue_analysis(
        sens_weight_mat,
        plot=True,
        save_plot=save_plot,
        fig_name="%s_network_connections.png" % list(NETWORK_TYPE.keys())[network_type]
    )


def main():
    # #################################################################################################################
    # Initialise parameters
    # #################################################################################################################
    verbosity = VERBOSITY
    num_neurons = int(1e4)
    save_plot = True
    networks = NETWORK_TYPE.keys()
    patches = 3
    c_alpha = 0.7
    compute_sum = False

    # #################################################################################################################
    # Parse command line arguments
    # #################################################################################################################
    cmd_params = arg_parse(sys.argv[1:])
    if cmd_params.seed:
        np.random.seed(0)

    if cmd_params.agg:
        import matplotlib
        matplotlib.use("Agg")

    if cmd_params.show:
        save_plot = False

    if cmd_params.network is not None:
        if cmd_params.network in list(NETWORK_TYPE.keys()):
            networks = [cmd_params.network]
        else:
            raise ValueError("Please pass a valid network as parameter")

    if cmd_params.verbosity is not None:
        verbosity = cmd_params.verbosity

    if cmd_params.num_neurons is not None:
        num_neurons = cmd_params.num_neurons

    if cmd_params.c_alpha is not None:
        c_alpha = cmd_params.c_alpha

    if cmd_params.patches is not None:
        patches = cmd_params.patches

    if cmd_params.sum:
        compute_sum = cmd_params.sum

    # #################################################################################################################
    # Run the eigenvalue spectra analysis
    # #################################################################################################################
    for network_type in networks:
        main_eigenvalue_spec(
            network_type=NETWORK_TYPE[network_type],
            num_neurons=num_neurons,
            patches=patches,
            c_alpha=c_alpha,
            compute_sum=compute_sum,
            save_plot=save_plot,
            verbosity=verbosity
        )


if __name__ == '__main__':
    main()


