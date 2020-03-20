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
from pathlib import Path
from webcolors import hex_to_rgb
import nest


VERBOSITY = 3
nest.set_verbosity("M_ERROR")


def main_lr(
        network_type=NETWORK_TYPE["local_circ_patchy_random"],
        input_type=INPUT_TYPE["plain"],
        reconstruct=True,
        tuning_function=TUNING_FUNCTION["gauss"],
        write_to_file=False,
        save_prefix='',
):
    """
    Main function to create a network, simulate and reconstruct the original stimulus
    :param network_type: The type of the network. This is an integer number defined in the NETWORK_TYPE dictionary
    :param input_type: The type of the input. This is an integer number defined in the INPUT_TYPE dictionary
    :param reconstruct: If set to true the stimulus is reconstructed. If set to False, the spatial variance is returned
    instead of the reconstruction
    :param tuning_function: The tuning function that is applied by the neurons. This is an integer number defined
    int the TUNING_FUNCTION dictionary
    :param write_to_file: If set to true the firing rate is written to an file
    :param save_prefix: Naming prefix that can be set before a file to mark a trial or an experiment
    :return: If reconstruct is set to False, the return values are the input stimulus, firing rates, and
    the averaged variance of the neighborhood. Otherwise the original input stimulus and the reconstructed stimulus
    is returned.
    """
    # load input stimulus
    input_stimulus = stimulus_factory(input_type)

    stimulus_fft = fourier_trans(input_stimulus)
    if VERBOSITY > 4:
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
    p_loc = 0.4
    p_lr = .1
    p_rf = 0.7
    pot_threshold = -55.
    pot_reset = -70.
    capacitance = 80.
    time_constant = 20.
    use_dc = False
    save_plots = write_to_file

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
        use_dc=use_dc,
        save_prefix=save_prefix,
        save_plots=save_plots,
        verbosity=VERBOSITY
    )
    network.create_network()

    if VERBOSITY > 2:
        print("\n#####################\tPlot in/out degree distribution")
        network.connect_distribution("connect_distribution.png")

    firing_rates, (spikes_s, time_s) = network.simulate(simulation_time)
    if write_to_file:
        fr_file = open("%s_firing_rates.txt" % save_prefix, "w+")
        fr_file.write(firing_rates)
        fr_file.close()

    if VERBOSITY > 0:
        average_firing_rate = np.mean(firing_rates)
        print("\n#####################\tAverage firing rate: %s" % average_firing_rate)

    if VERBOSITY > 2:
        print("\n#####################\tPlot firing pattern over time")
        plt.figure(figsize=(10, 5))
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
        if not save_plots:
            plt.show()
        else:
            curr_dir = os.getcwd()
            Path(curr_dir + "/figures/firing_rate").mkdir(parents=True, exist_ok=True)
            plt.savefig(curr_dir + "/figures/firing_rate/%s_firing_time.png" % save_prefix)

    if VERBOSITY > 2:
        print("\n#####################\tPlot firing pattern over space")
        plt.figure(figsize=(10, 5))
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
        if not save_plots:
            plt.show()
        else:
            curr_dir = os.getcwd()
            Path(curr_dir + "/figures/firing_rate").mkdir(parents=True, exist_ok=True)
            plt.savefig(curr_dir + "/figures/firing_rate/%s_firing_space.png" % save_prefix)

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
            _, fig = plt.subplots(1, 2, figsize=(10, 5))
            fig[0].imshow(np.abs(response_fft), norm=LogNorm(vmin=5))
            fig[1].imshow(np.abs(stimulus_fft), norm=LogNorm(vmin=5))
            if not save_plots:
                plt.show()
            else:
                curr_dir = os.getcwd()
                Path(curr_dir + "/figures/fourier").mkdir(parents=True, exist_ok=True)
                plt.savefig(curr_dir + "/figures/fourier/%s_fourier_trans.png" % save_prefix)

        if VERBOSITY > 1:
            _, fig_2 = plt.subplots(1, 2, figsize=(10, 5))
            fig_2[0].imshow(reconstruction, cmap='gray')
            fig_2[1].imshow(input_stimulus, cmap='gray', vmin=0, vmax=255)
            if not save_plots:
                plt.show()
            else:
                curr_dir = os.getcwd()
                Path(curr_dir + "/figures/reconstruction").mkdir(parents=True, exist_ok=True)
                plt.savefig(curr_dir + "/figures/reconstruction/%s_reconstruction.png" % save_prefix)

        return input_stimulus, reconstruction


def main_mi(input_type=INPUT_TYPE["plain"], num_trials=5):
    """
    Computes the mutual information that is averaged over several trials
    :param input_type: The input type. This is an integer number defined in the INPUT_TYPE dictionary
    :param num_trials: The number of trials that are conducted
    :return: None
    """
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


def main_error(
        input_type=INPUT_TYPE["plain"],
        network_type=NETWORK_TYPE["random"],
        tuning_function=TUNING_FUNCTION["gauss"],
        num_trials=5
):
    """
    Computes the average over the normalised L2 distance between the reconstructed and the original stimulus.
    :param input_type: The input type. This is an integer number defined in the INPUT_TYPE dictionary
    :param network_type: The network type. This is an integer number defined in the NETWORK_TYPE dictionary
    :param tuning_function: The tuning function. This is an integer number defined in the TUNING_FUNCTION dictionary
    :param num_trials: The number of trials
    :return: None
    """
    network_name = NETWORK_TYPE.keys()[network_type]
    input_name = INPUT_TYPE.keys()[input_type]
    tuning_name = TUNING_FUNCTION.keys()[network_type]

    errors = []
    for i in range(num_trials):
        input_stimulus, reconstruction = main_lr(
            network_type=network_type,
            input_type=input_type,
            reconstruct=True,
            tuning_function=tuning_function,
            write_to_file=True,
            save_prefix="error_%s_%s_%s_no_%s" % (network_name, input_name, tuning_name, i)
        )
        errors.append(error_distance(input_stimulus, reconstruction))

    mean_error = np.mean(np.asarray(errors))
    error_variance = np.var(np.asarray(errors))
    print("\n#####################\tMean Error for network type %s and input type %s: %s \n"
          % (network_name, input_name, mean_error))
    print("\n#####################\tError variance for network type %s and input type %s: %s \n"
          % (network_name, input_name, error_variance))


if __name__ == '__main__':
    cmd_params = arg_parse()
    experiment = None
    network_type = None
    input_type = None
    tuning_function = None

    if cmd_params.seed:
        np.random.seed(0)

    if cmd_params.agg:
        import matplotlib
        matplotlib.use("Agg")

    if cmd_params.experiment == "error":
        experiment = main_error
    elif cmd_params.experiment == "mi":
        experiment = main_mi
    else:
        raise ValueError("Please pass a valid experiment as parameter")

    if cmd_params.network in list(NETWORK_TYPE.keys()):
        network_type = NETWORK_TYPE[cmd_params.network]
    else:
        raise ValueError("Please pass a valid network as parameter")

    if cmd_params.input in list(INPUT_TYPE.keys()):
        input_type = NETWORK_TYPE[cmd_params.input]
    else:
        raise ValueError("Please pass a valid input type as parameter")

    # main_lr(network_type=NETWORK_TYPE["local_circ_patchy_sd"], input_type=INPUT_TYPE["perlin"], reconstruct=True)

    experiment(network_type=network_type, input_type=input_type, num_trials=10)

