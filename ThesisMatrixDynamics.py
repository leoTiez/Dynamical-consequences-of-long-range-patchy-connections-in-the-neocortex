#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.createStimulus import stimulus_factory, INPUT_TYPE
from modules.thesisUtils import arg_parse
from createThesisNetwork import network_factory, NETWORK_TYPE

import numpy as np
import matplotlib.pyplot as plt

import nest


VERBOSITY = 2
nest.set_verbosity("M_ERROR")


NETWORK_TYPE = {
    "random": 0,
    "local_circ": 1,
    "local_sd": 2,
    "local_circ_patchy_sd": 3,
    "local_circ_patchy_random": 4,
    "local_sd_patchy_sd": 5
}


def main_matrix_dynamics(network_type=NETWORK_TYPE["local_circ_patchy_sd"], input_type=INPUT_TYPE["plain"]):
    # load input stimulus
    stimulus_size = (50, 50)
    input_stimulus = stimulus_factory(input_type, size=stimulus_size)
    if VERBOSITY > 2:
        plt.imshow(input_stimulus, cmap='gray')
        plt.show()
    # #################################################################################################################
    # Define values
    # #################################################################################################################
    plot_arrangement_rows = 10
    plot_arrangement_columns = 10
    network_shape = (100, 100)
    num_neurons = int(network_shape[0] * network_shape[1])
    network = network_factory(input_stimulus, network_type=network_type, num_sensory=num_neurons, verbosity=VERBOSITY)
    network.create_network()
    sens_weight_mat = network.get_sensory_weight_mat()

    # Sufficient to use only 255 as we don't use the neurons themselves
    input_matrix = np.full(stimulus_size, 255)
    sensory_activity = network.adj_rec_sens_mat.T.dot(input_matrix.reshape(-1))
    fig, axes = plt.subplots(plot_arrangement_rows, plot_arrangement_columns)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    for ax in axes.reshape(-1):
        ax.imshow(sensory_activity.reshape(network_shape))
        sensory_activity = sens_weight_mat.T.dot(sensory_activity)

    plt.show()


if __name__ == '__main__':
    cmd_params = arg_parse()
    if cmd_params.seed:
        np.random.seed(0)
    if cmd_params.agg:
        import matplotlib
        matplotlib.use("Agg")

    main_matrix_dynamics(network_type=NETWORK_TYPE["local_circ_patchy_sd"], input_type=INPUT_TYPE["plain"])


