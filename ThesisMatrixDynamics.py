#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.createStimulus import stimulus_factory, INPUT_TYPE
from modules.thesisUtils import arg_parse
from createThesisNetwork import network_factory, NETWORK_TYPE

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path

import nest


VERBOSITY = 2
nest.set_verbosity("M_ERROR")


def main_matrix_dynamics(
        network_type=NETWORK_TYPE["local_circ_patchy_sd"],
        input_type=INPUT_TYPE["plain"],
        save_fig=True,
        save_prefix="",
        save_path=None
):
    """
    Calculate the network dynamics based on matrix calculations. For that, the input is once transformed via the
    feedforward matrix and then the dynamics are investigated via the weight matrix for the recurrent connections
    :param network_type: Network type. This is an integer number defined in the NETWORK_TYPE dictionary
    :param input_type: Input type. This is an integer number defined in INPUT_TYPE dictionary
    :param save_fig: If set to true, the created plot is saved
    :param save_prefix: The prefix that is used for the name of the saved plot
    :return: None
    """
    # load input stimulus
    stimulus_size = (50, 50)
    input_stimulus = stimulus_factory(input_type, size=stimulus_size)
    if VERBOSITY > 2:
        plt.imshow(input_stimulus, cmap='gray')
        plt.show()
    # #################################################################################################################
    # Define values
    # #################################################################################################################
    plot_arrangement_rows = 5
    plot_arrangement_columns = 5
    network_shape = (100, 100)
    num_neurons = int(network_shape[0] * network_shape[1])
    network = network_factory(input_stimulus, network_type=network_type, num_sensory=num_neurons, verbosity=VERBOSITY)
    network.create_network()
    sens_weight_mat = network.get_sensory_weight_mat()

    # Sufficient to use only 255 as we don't use the neurons themselves
    input_matrix = np.ones(stimulus_size[0] * stimulus_size[1] + 1)
    input_matrix[:stimulus_size[0] * stimulus_size[1]] = input_stimulus.reshape(-1)
    sensory_activity = input_matrix.reshape(-1).dot(network.ff_weight_mat)[:-1]
    fig, axes = plt.subplots(plot_arrangement_rows, plot_arrangement_columns, figsize=(20, 10))
    all_activities = []
    for ax in axes.reshape(-1):
        sensory_activity /= sensory_activity.max()
        all_activities.append(ax.imshow(sensory_activity.reshape(network_shape), cmap="cool"))
        sensory_activity = sens_weight_mat.T.dot(sensory_activity)

    fig.colorbar(all_activities[0], ax=axes, orientation='horizontal', fraction=.05)

    if not save_fig:
        plt.show()
    else:
        if save_path is None:
            curr_dir = os.getcwd()
            save_path = curr_dir + "/figures/matrix_dynamics"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig("%s/%s_matrix_dynamics.png" % (save_path, save_prefix))
        plt.close()


def main():
    cmd_params = arg_parse()
    if cmd_params.seed:
        np.random.seed(0)
    if cmd_params.agg:
        import matplotlib
        matplotlib.use("Agg")

    networks = NETWORK_TYPE.keys()
    stimuli = INPUT_TYPE.keys()
    save_fig = True

    if cmd_params.show:
        save_fig = False

    if cmd_params.network is not None:
        if cmd_params.network in list(NETWORK_TYPE.keys()):
            networks = [cmd_params.network]
        else:
            raise ValueError("Please pass a valid network as parameter")

    if cmd_params.input is not None:
        if cmd_params.input in list(INPUT_TYPE.keys()):
            stimuli = [cmd_params.input]
        else:
            raise ValueError("Please pass a valid input type as parameter")

    for net in networks:
        for stim in stimuli:
            save_prefix = "%s_%s" % (net, stim)
            main_matrix_dynamics(
                network_type=NETWORK_TYPE[net],
                input_type=INPUT_TYPE[stim],
                save_fig=save_fig,
                save_prefix=save_prefix
            )


if __name__ == '__main__':
    main()


