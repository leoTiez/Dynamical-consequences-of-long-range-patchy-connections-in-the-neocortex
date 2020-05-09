#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.createStimulus import stimulus_factory
from modules.thesisUtils import arg_parse
from createThesisNetwork import network_factory
from modules.thesisConstants import *

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path

import nest


VERBOSITY = 2
nest.set_verbosity("M_ERROR")


def main_matrix_dynamics(
        network_type="local_circ_patchy_sd",
        num_neurons=int(1e4),
        perlin_resolution=(4, 4),
        c_alpha=0.7,
        load_network=False,
        img_prop=0.4,
        normalise=False,
        save_fig=True,
        save_prefix="",
        save_path=None,
        verbosity=VERBOSITY
):
    """
    Calculate the network dynamics based on matrix calculations. For that, the input is once transformed via the
    feedforward matrix and then the dynamics are investigated via the weight matrix for the recurrent connections
    :param network_type: Network type. This is an integer number defined in the NETWORK_TYPE dictionary
    :param input_type: Input type. This is an integer number defined in INPUT_TYPE dictionary
    :param num_neurons: Number of sensory neurons. Needs to have an integer square root
    :param save_fig: If set to true, the created plot is saved
    :param save_prefix: The prefix that is used for the name of the saved plot
    :param save_path: Path which the figure is saved to. If save_fig is set to False this parameter is ignored
    :param verbosity: Sets the verbosity flag
    :return: None
    """
    # load input stimulus
    stimulus_size = (50, 50)
    input_stimulus = stimulus_factory(INPUT_TYPE["perlin"], resolution=perlin_resolution)
    if verbosity > 2:
        plt.imshow(input_stimulus, cmap='gray')
        plt.show()
    # #################################################################################################################
    # Define values
    # #################################################################################################################
    plot_arrangement_rows = 3
    plot_arrangement_columns = 4
    num_neurons = num_neurons
    network_shape = (int(np.sqrt(num_neurons)), int(np.sqrt(num_neurons)))
    network = network_factory(
        c_alpha=c_alpha,
        network_type=NETWORK_TYPE[network_type],
        img_prop=img_prop,
        num_sensory=num_neurons,
        verbosity=verbosity
    )
    if load_network:
        network.import_net(input_stimulus)
    else:
        network.create_network(input_stimulus)
    sens_weight_mat = network.get_sensory_weight_mat()

    # #################################################################################################################
    # Input stimulus propagation
    # #################################################################################################################
    # Sufficient to use only 255 as we don't use the neurons themselves
    fig, axes = plt.subplots(plot_arrangement_rows, plot_arrangement_columns, figsize=(20, 10))
    sensory_input = input_stimulus.reshape(-1).dot(network.ff_weight_mat)

    no_input_neurons = np.random.choice(
        network.input_neurons_mask.sum(),
        int(network.input_neurons_mask.sum() * (1 - img_prop)),
        replace=False
    )
    sensory_input[no_input_neurons] = 0
    sensory_activity = np.zeros(network.num_sensory)
    sensory_activity[network.input_neurons_mask] = sensory_input
    all_activities = []
    for ax in axes.reshape(-1):
        if normalise:
            sensory_activity -= sensory_activity.min()
            sensory_activity /= sensory_activity.max()
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        all_activities.append(ax.imshow(sensory_activity.reshape(network_shape), cmap="cool"))
        sensory_activity = sens_weight_mat.T.dot(sensory_activity)

    # #################################################################################################################
    # Plotting
    # #################################################################################################################
    if not normalise:
        vmin = min(image.get_array().min() for image in all_activities)
        vmax = max(image.get_array().max() for image in all_activities)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in all_activities:
            im.set_norm(norm)

    plt.rcParams.update({"font.size": 16})
    fig.colorbar(all_activities[0], ax=axes, orientation='horizontal', fraction=.05)

    title = ""
    if network_type == "random":
        title = "Random Network"
    elif network_type == "local_circ":
        title = "Circular Local Network"
    elif network_type == "local_sd":
        title = "Tuning Dependent Local Network"
    elif network_type == "local_circ_patchy_random":
        title = "Circular Local Network with Random Patches"
    elif network_type == "local_circ_patchy_sd":
        title = "Circular Local Network with Tuning Dependent Patches"
    else:
        title = "Tuning Dependent Local Network with Tuning Dependent Patches"

    plt.suptitle("Linear Dynamics for %s" % title)
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
    # #################################################################################################################
    # Initialise parameters
    # #################################################################################################################
    networks = list(NETWORK_TYPE.keys())[:-1]
    perlin_resolution = PERLIN_INPUT
    save_fig = True
    verbosity = VERBOSITY
    num_neurons = int(1e4)
    load_network = False
    c_alpha = 0.7
    normalise = False

    # #################################################################################################################
    # Parse and set command line arguments
    # #################################################################################################################
    cmd_params = arg_parse(sys.argv[1:])
    if cmd_params.seed:
        np.random.seed(0)

    if cmd_params.agg:
        import matplotlib
        matplotlib.use("Agg")

    if cmd_params.load_network:
        load_network = True

    if cmd_params.show:
        save_fig = False

    if cmd_params.normalise:
        normalise = True

    if cmd_params.network is not None:
        if cmd_params.network in list(NETWORK_TYPE.keys()):
            networks = [cmd_params.network]
        else:
            raise ValueError("Please pass a valid network as parameter")

    if cmd_params.c_alpha is not None:
        c_alpha = cmd_params.alpha

    if cmd_params.perlin is not None:
        perlin_resolution = [cmd_params.perlin]

    if cmd_params.verbosity is not None:
        verbosity = cmd_params.verbosity

    if cmd_params.num_neurons is not None:
        if int(np.sqrt(cmd_params.num_neurons))**2 == cmd_params.num_neurons:
            num_neurons = cmd_params.num_neurons
        else:
            raise ValueError("For plotting the network activity properly it is necessary to pass a number of neurons"
                             " whose square root is an integer value")

    # #################################################################################################################
    # Run stimulus propagation
    # #################################################################################################################
    for net in networks:
        for pr in perlin_resolution:
            save_prefix = "%s_%s" % (net, pr)
            main_matrix_dynamics(
                network_type=net,
                perlin_resolution=(pr, pr),
                num_neurons=num_neurons,
                c_alpha=c_alpha,
                normalise=normalise,
                save_fig=save_fig,
                save_prefix=save_prefix,
                verbosity=verbosity
            )


if __name__ == '__main__':
    main()


