#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.stimulusReconstruction import fourier_trans, direct_stimulus_reconstruction
from modules.createStimulus import *
from modules.thesisUtils import arg_parse
from createThesisNetwork import network_factory, NETWORK_TYPE
from modules.networkAnalysis import mutual_information_hist, error_distance

import numpy as np
import matplotlib.pyplot as plt
import nest


VERBOSITY = 1
nest.set_verbosity("M_ERROR")


def main_lr(network_type=NETWORK_TYPE["local_circ_patchy_random"], input_type=INPUT_TYPE["plain"], reconstruct=False):
    # load input stimulus
    input_stimulus = stimulus_factory(input_type)

    stimulus_fft = fourier_trans(input_stimulus)
    if VERBOSITY > 2:
        plt.imshow(input_stimulus, cmap='gray', vmin=0, vmax=255)
        plt.show()

    # #################################################################################################################
    # Define values
    # #################################################################################################################
    simulation_time = 1000.
    num_neurons = int(1e4)
    cap_s = 1.
    inh_weight = -15.
    all_same_input_current = False
    p_loc = 0.6
    p_lr = 0.2

    # Note: when using the same input current for all neurons, we obtain synchrony, and due to the refactory phase
    # all recurrent connections do not have any effect
    network = network_factory(
        input_stimulus,
        network_type=network_type,
        num_sensory=num_neurons,
        all_same_input_current=all_same_input_current,
        cap_s=cap_s,
        inh_weight=inh_weight,
        p_loc=p_loc,
        p_lr=p_lr,
        verbosity=VERBOSITY
    )
    network.create_network()
    firing_rates, (spikes_s, time_s) = network.simulate(simulation_time)

    if VERBOSITY > 0:
        average_firing_rate = np.mean(firing_rates)
        print("\n#####################\tAverage firing rate: %s" % average_firing_rate)

    if VERBOSITY > 3:
        print("\n#####################\tPlot firing pattern over time")
        positions = tp.GetPosition(spikes_s.tolist())
        plot_colorbar(plt.gcf(), plt.gca(), num_stim_classes=network.num_stim_discr)
        for s, t, pos in zip(spikes_s, time_s, positions):
            x_grid, y_grid = coordinates_to_cmap_index(network.layer_size, pos, network.spacing_perlin)
            stim_class = network.color_map[x_grid, y_grid]
            plt.plot(
                t,
                s,
                marker='.',
                markerfacecolor=list(mcolors.TABLEAU_COLORS.items())[stim_class][0]
                if s not in network.torus_inh_nodes else 'k',
                markeredgewidth=0
            )
        plt.show()

    if VERBOSITY > 2:
        print("\n#####################\tPlot firing pattern over space")
        plot_colorbar(plt.gcf(), plt.gca(), num_stim_classes=network.num_stim_discr)
        for pos, fr, neuron in zip(network.torus_layer_positions, firing_rates, network.torus_layer_nodes):
            if neuron not in network.torus_inh_nodes:
                x_grid, y_grid = coordinates_to_cmap_index(network.layer_size, pos, network.spacing_perlin)
                stim_class = network.color_map[x_grid, y_grid]
                plt.plot(
                    pos[0],
                    pos[1],
                    marker='o',
                    markerfacecolor=list(mcolors.TABLEAU_COLORS.items())[stim_class][0],
                    markeredgewidth=0,
                    alpha=fr/float(max(firing_rates))
                )
            else:
                plt.plot(
                    pos[0],
                    pos[1],
                    marker='o',
                    markerfacecolor='k',
                    markeredgewidth=0,
                    alpha=fr/float(max(firing_rates))
                )

        plt.imshow(
            network.color_map,
            cmap=custom_cmap(),
            alpha=0.3,
            origin=(network.color_map.shape[0] // 2, network.color_map.shape[1] // 2),
            extent=(
                -network.layer_size / 2.,
                network.layer_size / 2.,
                -network.layer_size / 2.,
                network.layer_size / 2.
            )
        )
        plt.show()

    if all_same_input_current or not reconstruct:
        return input_stimulus, firing_rates

    else:
        # #################################################################################################################
        # Reconstruct stimulus
        # #################################################################################################################
        # Reconstruct input stimulus
        if VERBOSITY > 0:
            print("\n#####################\tReconstruct stimulus")

        reconstruction = direct_stimulus_reconstruction(
            firing_rates,
            network.adj_rec_sens_mat,
            network.tuning_weight_vector
        )
        response_fft = fourier_trans(reconstruction)

        if VERBOSITY > 3:
            from matplotlib.colors import LogNorm
            _, fig = plt.subplots(1, 2)
            fig[0].imshow(np.abs(response_fft), norm=LogNorm(vmin=5))
            fig[1].imshow(np.abs(stimulus_fft), norm=LogNorm(vmin=5))

        if VERBOSITY > 1:
            _, fig_2 = plt.subplots(1, 3)
            fig_2[0].imshow(reconstruction, cmap='gray', vmin=0, vmax=255)
            fig_2[1].imshow(input_stimulus, cmap='gray', vmin=0, vmax=255)
            fig_2[2].imshow(network.color_map, cmap=custom_cmap())
            plt.show()

        return input_stimulus, reconstruction


def main_mi(input_type=INPUT_TYPE["plain"], num_trials=5):
    # Define parameters outside  the loop
    for network_type in list(NETWORK_TYPE.keys()):
        input_stimuli = []
        firing_rates = []
        for _ in range(num_trials):
            input_stimulus, firing_rate = main_lr(
                network_type=NETWORK_TYPE[network_type],
                input_type=input_type,
                reconstruct=False
            )
            input_stimuli.append(input_stimulus.reshape(-1))
            firing_rates.append(firing_rate.reshape(-1))

        mutual_information = mutual_information_hist(input_stimuli, firing_rates)
        print("\n#####################\tMutual Information MI for network type %s and input type %s: %s \n"
              % (network_type, input_type, mutual_information))


def main_error(input_type=INPUT_TYPE["plain"], num_trials=5):
    for network_type in list(NETWORK_TYPE.keys()):
        errors = []
        for _ in range(num_trials):
            input_stimulus, reconstruction = main_lr(
                network_type=NETWORK_TYPE[network_type],
                input_type=input_type,
                reconstruct=True
            )
            errors.append(error_distance(input_stimulus, reconstruction))

        mean_error = np.mean(np.asarray(errors))
        print("\n#####################\tMean Error for network type %s and input type %s: %s \n"
              % (network_type, input_type, mean_error))


if __name__ == '__main__':
    cmd_params = arg_parse()
    if cmd_params.seed:
        np.random.seed(0)
    if cmd_params.agg:
        import matplotlib
        matplotlib.use("Agg")

    main_lr(network_type=NETWORK_TYPE["local_circ_patchy_sd"], input_type=INPUT_TYPE["perlin"])
    # main_mi(input_type=INPUT_TYPE["perlin"], num_trials=3)
    # main_error(input_type=INPUT_TYPE["plain"], num_trials=5)

