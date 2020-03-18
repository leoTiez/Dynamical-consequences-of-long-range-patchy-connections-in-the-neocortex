#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.stimulusReconstruction import fourier_trans, direct_stimulus_reconstruction
from modules.createStimulus import *
from modules.thesisUtils import arg_parse
from modules.networkConstruction import TUNING_FUNCTION
from createThesisNetwork import network_factory, NETWORK_TYPE
from modules.networkAnalysis import mutual_information_hist, error_distance, spatial_variance

import numpy as np
import matplotlib.pyplot as plt
from webcolors import hex_to_rgb
import nest


VERBOSITY = 4
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
    p_loc = 0.5
    p_lr = .2
    p_rf = 0.7
    pot_threshold = -55.
    pot_reset = -70.
    capacitance = 80.
    time_constant = 20.
    tuning_function = TUNING_FUNCTION["gauss"]

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
        p_rf=p_rf,
        pot_reset=pot_reset,
        pot_threshold=pot_threshold,
        capacitance=capacitance,
        time_constant=time_constant,
        tuning_function=tuning_function,
        verbosity=VERBOSITY
    )
    network.create_network()
    firing_rates, (spikes_s, time_s) = network.simulate(simulation_time)

    if VERBOSITY > 0:
        average_firing_rate = np.mean(firing_rates)
        print("\n#####################\tAverage firing rate: %s" % average_firing_rate)

    if VERBOSITY > 2:
        print("\n#####################\tPlot firing pattern over time")
        positions = np.asarray(tp.GetPosition(spikes_s.tolist()))
        plot_colorbar(plt.gcf(), plt.gca(), num_stim_classes=network.num_stim_discr)

        inh_mask = np.zeros(len(spikes_s)).astype('bool')
        for inh_n in network.torus_inh_nodes:
            inh_mask[spikes_s == inh_n] = True

        x_grid, y_grid = coordinates_to_cmap_index(network.layer_size, positions[~inh_mask], network.spacing_perlin)
        stim_classes = network.color_map[x_grid, y_grid]
        c = np.full(len(spikes_s), '#000000')
        c[~inh_mask] = np.asarray(list(mcolors.TABLEAU_COLORS.items()))[stim_classes, 1]
        plt.scatter(time_s, spikes_s, c=c.tolist(), marker=',')
        plt.show()

    if VERBOSITY > 2:
        print("\n#####################\tPlot firing pattern over space")
        plot_colorbar(plt.gcf(), plt.gca(), num_stim_classes=network.num_stim_discr)

        inh_mask = np.zeros(len(network.torus_layer_nodes)).astype('bool')
        inh_mask[np.asarray(network.torus_inh_nodes) - min(network.torus_layer_nodes)] = True

        x_grid, y_grid = coordinates_to_cmap_index(
            network.layer_size,
            np.asarray(network.torus_layer_positions)[~inh_mask],
            network.spacing_perlin
        )
        stim_classes = network.color_map[x_grid, y_grid]

        c = np.full(len(network.torus_layer_nodes), '#000000')
        c[~inh_mask] = np.asarray(list(mcolors.TABLEAU_COLORS.items()))[stim_classes, 1]

        c_rgba = np.zeros((len(network.torus_layer_nodes), 4))
        for num, color in enumerate(c):
            c_rgba[num, :3] = np.asarray(hex_to_rgb(color))[:] / 255.
        c_rgba[:, 3] = firing_rates/float(max(firing_rates))
        plt.scatter(
            np.asarray(network.torus_layer_positions)[:, 0],
            np.asarray(network.torus_layer_positions)[:, 1],
            c=c_rgba
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

    mean_variance = spatial_variance(network.torus_layer_tree, network.torus_layer_positions, firing_rates)
    if all_same_input_current or not reconstruct:
        return input_stimulus, firing_rates, mean_variance

    else:
        # #################################################################################################################
        # Reconstruct stimulus
        # #################################################################################################################
        # Reconstruct input stimulus
        if VERBOSITY > 0:
            print("\n#####################\tReconstruct stimulus")

        reconstruction = direct_stimulus_reconstruction(
            firing_rates,
            network.ff_weight_mat,
        )
        response_fft = fourier_trans(reconstruction)

        if VERBOSITY > 3:
            from matplotlib.colors import LogNorm
            _, fig = plt.subplots(1, 2)
            fig[0].imshow(np.abs(response_fft), norm=LogNorm(vmin=5))
            fig[1].imshow(np.abs(stimulus_fft), norm=LogNorm(vmin=5))

        if VERBOSITY > 1:
            _, fig_2 = plt.subplots(1, 2)
            fig_2[0].imshow(reconstruction, cmap='gray')
            fig_2[1].imshow(input_stimulus, cmap='gray', vmin=0, vmax=255)
            plt.show()

        return input_stimulus, reconstruction


def main_mi(input_type=INPUT_TYPE["plain"], num_trials=5):
    # Define parameters outside  the loop
    for network_type in list(NETWORK_TYPE.keys()):
        input_stimuli = []
        firing_rates = []
        variance = []
        for _ in range(num_trials):
            input_stimulus, firing_rate, corr = main_lr(
                network_type=NETWORK_TYPE[network_type],
                input_type=input_type,
                reconstruct=False
            )
            input_stimuli.append(input_stimulus.reshape(-1))
            firing_rates.append(firing_rate.reshape(-1))
            variance.append(corr)

        mutual_information = mutual_information_hist(input_stimuli, firing_rates)
        print("\n#####################\tMutual Information MI for network type %s and input type %s: %s \n"
              % (network_type, input_type, mutual_information))
        print("\n#####################\tSpatial variance for network type %s and input type %s: %s \n"
              % (network_type, input_type, np.asarray(variance).mean()))


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

    main_lr(network_type=NETWORK_TYPE["local_circ_patchy_sd"], input_type=INPUT_TYPE["perlin"], reconstruct=True)
    # main_mi(input_type=INPUT_TYPE["perlin"], num_trials=3)
    # main_error(input_type=INPUT_TYPE["plain"], num_trials=5)

