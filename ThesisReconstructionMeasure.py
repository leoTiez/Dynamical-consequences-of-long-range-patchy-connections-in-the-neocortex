#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.stimulusReconstruction import fourier_trans, direct_stimulus_reconstruction
from modules.createStimulus import *
from modules.thesisUtils import arg_parse, firing_rate_sorting
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


PARAMETER_DICT = {
    "tuning": 0,
    "cluster": 1,
    "patches": 2,
    "perlin": 3,
    "weights": 4
}


def main_lr(
        network_type=NETWORK_TYPE["local_circ_patchy_random"],
        input_type=INPUT_TYPE["plain"],
        cluster=(15, 15),
        tuning_function=TUNING_FUNCTION["gauss"],
        perlin_input_cluster=(5, 5),
        num_patches=3,
        weight_factor=(1., 1.),
        img_prop=1.,
        spatial_sampling=False,
        write_to_file=False,
        save_plots=True,
        save_prefix='',
):
    """
    Main function to create a network, simulate and reconstruct the original stimulus
    :param network_type: The type of the network. This is an integer number defined in the NETWORK_TYPE dictionary
    :param input_type: The type of the input. This is an integer number defined in the INPUT_TYPE dictionary
    :param cluster: The size of the Perlin noise mesh
    :param tuning_function: The tuning function that is applied by the neurons. This is an integer number defined
    int the TUNING_FUNCTION dictionary
    :param perlin_input_cluster: Cluster size of the perlin input image. If the input is not perlin, this parameter
    is ignored
    :param num_patches: number of patches. If the network does not establish patches this parameter is ignored
    :param img_prop: Proportion of the image information that is used
    :param spatial_sampling: If set to true, the neurons that receive ff input are chosen with spatial correlation
    :param write_to_file: If set to true the firing rate is written to an file
    :param save_plots: If set to true, plots are saved instead of being displayed
    :param save_prefix: Naming prefix that can be set before a file to mark a trial or an experiment
    :return: The original image, the reconstructed image and the firing rates
    """
    # load input stimulus
    input_stimulus = stimulus_factory(input_type, resolution=perlin_input_cluster)

    stimulus_fft = fourier_trans(input_stimulus)
    if VERBOSITY > 2:
        if not write_to_file:
            plt.imshow(input_stimulus, cmap='gray', vmin=0, vmax=255)
            plt.show()
        else:
            curr_dir = os.getcwd()
            Path(curr_dir + "/figures/input").mkdir(parents=True, exist_ok=True)
            plt.savefig(curr_dir + "/figures/input/%s_input.png" % save_prefix)
            plt.close()
            
    # #################################################################################################################
    # Define values
    # #################################################################################################################
    simulation_time = 1000.
    num_neurons = int(1e4)
    cap_s = 1. * weight_factor[0]
    inh_weight = -15. * weight_factor[0] ** weight_factor[0]
    ff_weight = 1.0 * weight_factor[1]
    all_same_input_current = False
    p_loc = 0.4
    p_lr = .1
    p_rf = 0.7
    pot_threshold = -55.
    pot_reset = -70.
    capacitance = 80.
    time_constant = 20.
    use_dc = False

    # Note: when using the same input current for all neurons, we obtain synchrony, and due to the refactory phase
    # all recurrent connections do not have any effect
    network = network_factory(
        input_stimulus,
        network_type=network_type,
        num_sensory=num_neurons,
        all_same_input_current=all_same_input_current,
        ff_weight=ff_weight,
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
        resolution_perlin=cluster,
        num_patches=num_patches,
        use_input_neurons=True if network_type == NETWORK_TYPE["input_only"] else False,
        img_prop=img_prop,
        spatial_sampling=spatial_sampling,
        use_dc=use_dc,
        save_prefix=save_prefix,
        save_plots=save_plots,
        verbosity=VERBOSITY
    )
    network.create_network()

    if VERBOSITY > 3:
        print("\n#####################\tPlot in/out degree distribution")
        network.connect_distribution("connect_distribution.png")

    if network_type == NETWORK_TYPE["input_only"]:
        reconstruction = network.input_recon
        firing_rates = np.zeros(network.num_sensory)
        return input_stimulus, reconstruction, firing_rates

    firing_rates, (spikes_s, time_s) = network.simulate(simulation_time)
    if write_to_file:
        curr_dir = os.getcwd()
        Path(curr_dir + "/firing_rates_files/").mkdir(exist_ok=True, parents=True)
        fr_file = open(curr_dir + "/firing_rates_files/%s_firing_rates.txt" % save_prefix, "w+")
        fr_file.write(str(firing_rates.tolist()))
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
        cl = np.full(len(spikes_s), -1)
        cl[~inh_mask] = stim_classes
        c = np.full(len(spikes_s), '#000000')
        c[~inh_mask] = np.asarray(list(mcolors.TABLEAU_COLORS.items()))[stim_classes, 1]
        sorted_zip = sorted(zip(time_s, spikes_s, c, cl), key=lambda l: l[3])
        sorted_time, sorted_spikes, sorted_c, _ = zip(*sorted_zip)
        new_idx_spikes = []
        new_idx_neurons = {}
        for s in sorted_spikes:
            new_idx_spikes.append(firing_rate_sorting(new_idx_spikes, sorted_spikes, new_idx_neurons, s))
        plt.scatter(sorted_time, new_idx_spikes, c=list(sorted_c), marker='.')
        if not save_plots:
            plt.show()
        else:
            curr_dir = os.getcwd()
            Path(curr_dir + "/figures/firing_rate").mkdir(parents=True, exist_ok=True)
            plt.savefig(curr_dir + "/figures/firing_rate/%s_firing_time.png" % save_prefix)
            plt.close()

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
            plt.close()

    # #############################################################################################################
    # Reconstruct stimulus
    # #############################################################################################################
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
            plt.close()

    if VERBOSITY > 1:
        plot_reconstruction(input_stimulus, reconstruction, save_plots=save_plots, save_prefix=save_prefix)

    return input_stimulus, reconstruction, firing_rates


def experiment(
        input_type=INPUT_TYPE["plain"],
        network_type=NETWORK_TYPE["random"],
        tuning_function=TUNING_FUNCTION["gauss"],
        cluster=(15, 15),
        perlin_input_cluster=(5, 5),
        patches=3,
        weight_factor=(1., 1.),
        img_prop=1.,
        spatial_sampling=False,
        save_plots=True,
        num_trials=10
):
    """
    Computes the mutual information that is averaged over several trials
    :param input_type: The input type. This is an integer number defined in the INPUT_TYPE dictionary
    :param network_type: The network type. This is an integer number defined in the NETWORK_TYPE dictionary
    :param tuning_function: The tuning function of senory neurons. This is an integer number defined in the
    TUNING_FUNCTION dictionary
    :param cluster: The size of the mesh that is used for the Perlin noise distribution of the sensory neurons
    The parameter is ignored if random network is chosen
    :param perlin_input_cluster: Cluster size of the perlin input image
    :param patches: The number of patches. This parameter is ignored if network is chosen that does not make use of
    patchy connctions
    :param img_prop: Defines the sparse sampling, i.e. the number of neurons that receive feedforward input.
    :param spatial_sampling: If set to true, the neurons that receive ff input are chosen with spatial correlation
    :param save_plots: If set to true, plots are saved instead of being displayed
    :param num_trials: The number of trials that are conducted
    :return: None
    """
    network_name = list(NETWORK_TYPE.keys())[network_type]
    input_name = list(INPUT_TYPE.keys())[input_type]
    parameters = [cluster, patches, num_trials]
    if sum(1 for _ in filter(None.__ne__, parameters)) < len(parameters) - 1:
        raise ValueError("The experiment cannot change more than one parameter at a time")

    parameters = []
    parameter_str = ""
    if tuning_function is None:
        parameters = TUNING_FUNCTION.values()
        parameter_str = "tuning_function"
    elif cluster is None:
        parameters = [(4, 4), (8, 8), (12, 12), (16, 16), (20, 20)]
        parameter_str = "orientation_map"
    elif patches is None:
        parameters = np.arange(1, 5, 1)
        parameter_str = "num_patches"
    elif perlin_input_cluster is None:
        parameters = [(8, 8), (15, 15), (20, 20)]
        parameter_str = "perlin_cluster_size"
    elif weight_factor is None:
        parameters = [(2., 0.9), (5., 0.8), (7., 0.7), (10., 0.6)]
        parameter_str = "weight_balance"

    if len(list(parameters)) == 0:
        parameters.append("")

    curr_dir = os.getcwd()
    Path(curr_dir + "/error/").mkdir(exist_ok=True, parents=True)
    Path(curr_dir + "/mi/").mkdir(exist_ok=True, parents=True)

    for p in parameters:
        input_stimuli = []
        firing_rates = []
        errors = []
        tuning_name = list(TUNING_FUNCTION.keys())[p if tuning_function is None else tuning_function]
        for i in range(num_trials):
            save_prefix = "%s_%s_%s_%s_img_prop_%s_no_%s" % (
                network_name,
                input_name,
                parameter_str,
                p,
                img_prop,
                i
            )
            if VERBOSITY > 0:
                print("\n#####################\tThe save prefix is: ", save_prefix)

            input_stimulus, reconstruction, firing_rate = main_lr(
                network_type=network_type,
                input_type=input_type,
                tuning_function=p if tuning_function is None else tuning_function,
                cluster=p if cluster is None else cluster,
                num_patches=p if patches is None else patches,
                perlin_input_cluster=p if perlin_input_cluster is None else perlin_input_cluster,
                weight_factor=p if weight_factor is None else weight_factor,
                img_prop=img_prop,
                spatial_sampling=spatial_sampling,
                write_to_file=True,
                save_plots=save_plots,
                save_prefix=save_prefix
            )

            ed = error_distance(input_stimulus, reconstruction)
            ed_file = open(curr_dir + "/error/%s_error_distance.txt" % save_prefix, "w+")
            ed_file.write(str(ed))
            ed_file.close()

            errors.append(ed)
            input_stimuli.append(input_stimulus.reshape(-1))
            firing_rates.append(firing_rate.reshape(-1))

        save_prefix = "%s_%s_%s_%s_img_prop_%s" % (
            network_name,
            input_name,
            parameter_str,
            p if tuning_function is not None else tuning_name,
            img_prop
        )

        mean_error = np.mean(np.asarray(errors))
        error_variance = np.var(np.asarray(errors))
        mutual_information = mutual_information_hist(input_stimuli, firing_rates)

        mean_error_file = open(curr_dir + "/error/%s_mean_error.txt" % save_prefix, "w+")
        mean_error_file.write(str(mean_error))
        mean_error_file.close()

        error_variance_file = open(curr_dir + "/error/%s_error_variance.txt" % save_prefix, "w+")
        error_variance_file.write(str(error_variance))
        error_variance_file.close()

        mi_file = open(curr_dir + "/mi/%s_mi.txt" % save_prefix, "w+")
        mi_file.write(str(mutual_information))
        mi_file.close()

        if VERBOSITY > 0:
            print("\n#####################\tMean Error for network type %s, %s %s, image proportion %s,"
                  " and input type %s: %s \n"
                  % (
                      network_name,
                      parameter_str,
                      p if tuning_function is not None else tuning_name,
                      img_prop,
                      input_name,
                      mean_error
                  ))
            print("\n#####################\tError variance for network type %s, %s %s, image proportion %s,"
                  " and input type %s: %s \n"
                  % (
                      network_name,
                      parameter_str,
                      p if tuning_function is not None else tuning_name,
                      img_prop,
                      input_name,
                      error_variance
                  ))
            print("\n#####################\tMutual Information MI for network type %s, %s %s, image proportion %s,"
                  " and input type %s: %s \n"
                  % (
                      network_name,
                      parameter_str,
                      p if tuning_function is not None else tuning_name,
                      img_prop,
                      input_name,
                      mutual_information
                  ))


if __name__ == '__main__':
    cmd_params = arg_parse()
    network_type = None
    input_type = None
    tuning_function = TUNING_FUNCTION["gauss"]
    cluster = (15, 15)
    perlin_input_cluster = (5, 5)
    num_trials = 10
    patches = 3
    weight_factor = (1., 1.)
    img_prop = 1.
    spatial_sampling = False
    save_plots = True

    if cmd_params.seed:
        np.random.seed(0)

    if cmd_params.agg:
        import matplotlib
        matplotlib.use("Agg")

    if cmd_params.spatial_sampling:
        spatial_sampling = True

    if cmd_params.show:
        save_plots = False

    if cmd_params.network in list(NETWORK_TYPE.keys()):
        network_type = NETWORK_TYPE[cmd_params.network]
    else:
        raise ValueError("Please pass a valid network as parameter")

    if cmd_params.input in list(INPUT_TYPE.keys()):
        input_type = INPUT_TYPE[cmd_params.input]
    else:
        raise ValueError("Please pass a valid input type as parameter")

    if cmd_params.parameter in list(PARAMETER_DICT.keys()):
        if cmd_params.parameter.lower() == "tuning":
            tuning_function = None
        elif cmd_params.parameter.lower() == "patches":
            if "patchy" not in cmd_params.network.lower():
                raise ValueError("Cannot run experiments about the number of patches a non-patchy network")
            patches = None
        elif cmd_params.parameter.lower() == "cluster":
            if network_type == NETWORK_TYPE["random"]:
                raise ValueError("Cannot run experiments about the cluster size with a random network")
            cluster = None
        elif cmd_params.parameter.lower() == "perlin":
            if cmd_params.input is not None:
                if cmd_params.input.lower() != "perlin":
                    raise ValueError("Cannot investigate the effect of the perlin size when not using perlin as input")
            perlin_input_cluster = None
        elif cmd_params.parameter.lower() == "weights":
            weight_factor= None

    if cmd_params.tuning is not None:
        tuning_function = TUNING_FUNCTION[cmd_params.tuning]

    if cmd_params.cluster is not None:
        cluster = cmd_params.cluster

    if cmd_params.patches is not None:
        patches = cmd_params.patches

    if cmd_params.num_trials is not None:
        num_trials = cmd_params.num_trials

    if cmd_params.weight_factor is not None:
        weight_factor = cmd_params.weight_factor

    if cmd_params.img_prop is not None:
        img_prop = float(cmd_params.img_prop)

    # main_lr(
    #     network_type=NETWORK_TYPE["local_circ_patchy_sd"],
    #     input_type=INPUT_TYPE["perlin"],
    #     tuning_function=TUNING_FUNCTION["step"],
    #     img_prop=1.,
    # )

    print("Start experiments for network %s given the input %s."
          " The parameter %s is changed."
          " The number of trials is %s"
          " and sampling rate is %s with%s spatial correlation"
          % (
              cmd_params.network,
              cmd_params.input,
              cmd_params.parameter,
              num_trials,
              img_prop,
              "" if spatial_sampling else "out"
          ))

    experiment(
        network_type=network_type,
        input_type=input_type,
        tuning_function=tuning_function,
        cluster=cluster,
        perlin_input_cluster=perlin_input_cluster,
        patches=patches,
        weight_factor=weight_factor,
        img_prop=img_prop,
        spatial_sampling=spatial_sampling,
        save_plots=save_plots,
        num_trials=num_trials
    )

