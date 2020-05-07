#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.stimulusReconstruction import fourier_trans, oblivious_stimulus_reconstruction
from modules.createStimulus import stimulus_factory
from modules.thesisUtils import *
from createThesisNetwork import network_factory
from modules.networkAnalysis import error_distance
from modules.thesisConstants import *

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nest


VERBOSITY = 3
nest.set_verbosity("M_ERROR")


def main_lr(
        network_type=NETWORK_TYPE["local_circ_patchy_random"],
        num_neurons=int(1e4),
        cluster=(15, 15),
        tuning_function=TUNING_FUNCTION["gauss"],
        perlin_input_cluster=(4, 4),
        num_patches=3,
        rec_factor=1.,
        c_alpha=0.7,
        img_prop=1.,
        presentation_time=1000.,
        spatial_sampling=False,
        use_equilibrium=False,
        load_network=False,
        write_to_file=False,
        save_plots=True,
        save_prefix='',
        verbosity=VERBOSITY
):
    """
    Main function to create a network, simulate and reconstruct the original stimulus
    :param network_type: The type of the network. This is an integer number defined in the NETWORK_TYPE dictionary
    :param num_neurons: Number of sensory neurons
    :param cluster: The size of the Perlin noise mesh
    :param tuning_function: The tuning function that is applied by the neurons. This is an integer number defined
    int the TUNING_FUNCTION dictionary
    :param perlin_input_cluster: Cluster size of the perlin input image. If the input is not perlin, this parameter
    is ignored
    :param num_patches: number of patches. If the network does not establish patches this parameter is ignored
    :param rec_factor: Multiplier for the ff weights
    :param img_prop: Proportion of the image information that is used
    :param spatial_sampling: If set to true, the neurons that receive ff input are chosen with spatial correlation
    :param use_equilibrium: If set to true, only the last 400ms of the simulation are used, ie when the network is
    expected to approach equilibrium
    :param write_to_file: If set to true the firing rate is written to an file
    :param save_plots: If set to true, plots are saved instead of being displayed
    :param save_prefix: Naming prefix that can be set before a file to mark a trial or an experiment
    :param verbosity: Verbosity flag
    :return: The original image, the reconstructed image and the firing rates
    """
    # #################################################################################################################
    # Load stimulus
    # #################################################################################################################
    input_stimulus = stimulus_factory(INPUT_TYPE["perlin"], resolution=perlin_input_cluster)

    stimulus_fft = fourier_trans(input_stimulus)
    if verbosity > 2:
        plt.imshow(input_stimulus, origin="lower", cmap="gray", vmin=0, vmax=255)
        if not save_plots:
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
    eq_time = 600.
    cap_s = 1.
    inh_weight = -5.
    ff_weight = 1.0
    max_spiking = 2000.
    bg_rate = 500.
    all_same_input_current = False
    p_rf = 0.7
    pot_threshold = -55.
    pot_reset = -70.
    capacitance = 80.
    time_constant = 20.
    use_dc = False

    plot_start = 1.
    plot_end = 200.
    min_mem_pot = 10.

    # #################################################################################################################
    # Create network
    # #################################################################################################################
    # Note: when using the same input current for all neurons, we obtain synchrony, and due to the refactory phase
    # all recurrent connections do not have any effect
    network = network_factory(
        network_type=network_type,
        num_sensory=num_neurons,
        all_same_input_current=all_same_input_current,
        ff_weight=ff_weight,
        cap_s=cap_s,
        inh_weight=inh_weight,
        c_alpha=c_alpha,
        p_rf=p_rf,
        bg_rate=bg_rate,
        max_spiking=max_spiking,
        rec_factor=rec_factor,
        pot_reset=pot_reset,
        pot_threshold=pot_threshold,
        capacitance=capacitance,
        time_constant=time_constant,
        tuning_function=tuning_function,
        presentation_time=presentation_time,
        resolution_perlin=cluster,
        num_patches=num_patches,
        use_input_neurons=True if network_type == NETWORK_TYPE["input_only"] else False,
        img_prop=img_prop,
        spatial_sampling=spatial_sampling,
        use_dc=use_dc,
        save_prefix=save_prefix,
        save_plots=save_plots,
        verbosity=verbosity,
        to_file=write_to_file
    )
    if load_network:
        print_msg("Import network")
        network.import_net(input_stimulus)
    else:
        network.create_network(input_stimulus)

    if verbosity > 4:
        print_msg("Plot in/out degree distribution")
        network.connect_distribution(distinguish_connections=False, plot_name="connect_distribution.png")

    if network_type == NETWORK_TYPE["input_only"]:
        reconstruction = network.input_recon
        firing_rates = np.zeros(network.num_sensory)
        return input_stimulus, reconstruction, firing_rates

    # #################################################################################################################
    # Simulate
    # #################################################################################################################
    firing_rates, (spikes_s, time_s) = network.simulate(
        simulation_time,
        use_equilibrium=use_equilibrium,
        eq_time=eq_time
    )

    if write_to_file:
        curr_dir = os.getcwd()
        Path(curr_dir + "/firing_rates_files/").mkdir(exist_ok=True, parents=True)
        fr_file = open(curr_dir + "/firing_rates_files/%s_firing_rates.txt" % save_prefix, "w+")
        fr_file.write(str(firing_rates.tolist()))
        fr_file.close()

    if verbosity > 0:
        average_firing_rate = np.mean(firing_rates)
        print_msg("Average firing rate: %s" % average_firing_rate)

    # #################################################################################################################
    # Plot neural activity
    # #################################################################################################################
    if verbosity > 2:
        print_msg("Plot firing pattern over time")
        plot_spikes_over_time(
            spikes_s,
            time_s,
            network,
            t_start=0.,
            t_end=simulation_time,
            t_stim_start=[eq_time] if use_equilibrium else [],
            t_stim_end=[presentation_time] if presentation_time < simulation_time else [],
            save_plot=save_plots,
            save_prefix=save_prefix
        )

    c_rgba = None
    if verbosity > 2:
        print_msg("Plot firing pattern over space")
        c_rgba = plot_spikes_over_space(firing_rates, network, save_plot=save_plots, save_prefix=save_prefix)

    if verbosity > 4:
        print_msg("Create network animation")
        plot_network_animation(
            network,
            spikes_s,
            time_s,
            c_rgba=c_rgba,
            min_mem_pot=min_mem_pot,
            animation_start=plot_start,
            animation_end=plot_end,
            save_plot=save_plots,
            save_prefix=save_prefix
        )

    # #############################################################################################################
    # Reconstruct stimulus
    # #############################################################################################################
    # Reconstruct input stimulus
    if verbosity > 0:
        print_msg("Reconstruct stimulus")

    reconstruction = oblivious_stimulus_reconstruction(
        firing_rates,
        network.input_neurons_mask,
        network.ff_weight_mat,
        network.tuning_vector
    )
    response_fft = fourier_trans(reconstruction)

    if verbosity > 3:
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

    if verbosity > 1:
        plot_reconstruction(input_stimulus, reconstruction, save_plots=save_plots, save_prefix=save_prefix)

    return input_stimulus, reconstruction, firing_rates


def experiment(
        network_type=NETWORK_TYPE["random"],
        num_neurons=int(1e4),
        tuning_function=TUNING_FUNCTION["gauss"],
        cluster=(15, 15),
        perlin_input_cluster=(4, 4),
        patches=3,
        rec_factor=1.,
        c_alpha=0.7,
        img_prop=1.,
        presentation_time=1000.,
        spatial_sampling=False,
        use_equilibrium=False,
        load_network=False,
        existing_ok=False,
        save_plots=True,
        num_trials=10,
        verbosity=VERBOSITY
):
    """
    Computes the mutual information that is averaged over several trials
    :param network_type: The network type. This is an integer number defined in the NETWORK_TYPE dictionary
    :param num_neurons: Set the number of sensory neurons
    :param tuning_function: The tuning function of senory neurons. This is an integer number defined in the
    TUNING_FUNCTION dictionary
    :param cluster: The size of the mesh that is used for the Perlin noise distribution of the sensory neurons
    The parameter is ignored if random network is chosen
    :param perlin_input_cluster: Cluster size of the perlin input image
    :param patches: The number of patches. This parameter is ignored if network is chosen that does not make use of
    patchy connctions
    :param rec_factor: Multiplier for the ff weights
    (second index)
    :param img_prop: Defines the sparse sampling, i.e. the number of neurons that receive feedforward input.
    :param spatial_sampling: If set to true, the neurons that receive ff input are chosen with spatial correlation
    :param use_equilibrium: If set to true, only the last 400ms of the simulation is used, ie when the network
    is expected to approach equilibrium
    :param save_plots: If set to true, plots are saved instead of being displayed
    :param num_trials: The number of trials that are conducted
    :param verbosity: Set the verbosity flag
    :return: None
    """
    # #################################################################################################################
    # Set experiment parameters
    # #################################################################################################################
    network_name = list(NETWORK_TYPE.keys())[network_type]
    input_name = str(perlin_input_cluster[0])
    parameters = [tuning_function, cluster, patches, rec_factor, c_alpha]
    if sum(1 for _ in filter(None.__ne__, parameters)) < len(parameters) - 1:
        raise ValueError("The experiment cannot change more than one parameter at a time")

    parameters = []
    parameter_str = ""
    if tuning_function is None:
        parameters = TUNING_FUNCTION.values()
        parameter_str = "tuning_function"
    elif cluster is None:
        parameters = FUNC_MAP_CLUSTER_PAR
        parameter_str = "orientation_map"
    elif patches is None:
        parameters = PATCHES_PAR
        parameter_str = "num_patches"
        load_network = False
    elif rec_factor is None:
        parameters = REC_FACTORS_PAR
        parameter_str = "weight_balance"
    elif c_alpha is None:
        parameters = ALPHA_PAR
        parameter_str = "c_alpha"

    if len(list(parameters)) == 0:
        parameters.append("")

    curr_dir = os.getcwd()
    Path(curr_dir + "/error/").mkdir(exist_ok=True, parents=True)

    # #################################################################################################################
    # Loop over parameter range
    # #################################################################################################################
    for p in parameters:
        input_stimuli = []
        firing_rates = []
        errors = []
        tuning_name = list(TUNING_FUNCTION.keys())[p if tuning_function is None else tuning_function]

        start_index = 0
        save_prefix_root = "%s_%s_%s_%s_img_prop_%s_spatials_%s" % (
            network_name,
            input_name,
            parameter_str,
            p,
            img_prop,
            spatial_sampling
        )
        if existing_ok:
            files = os.listdir(curr_dir + "/error/")
            files = [f for f in files if save_prefix_root in f]
            start_index = np.minimum(num_trials, len(files))

        for i in range(start_index, num_trials):
            save_prefix = "%s_no_%s" % (save_prefix_root, i)
            if verbosity > 0:
                print_msg("The save prefix is: %s" % save_prefix)

            input_stimulus, reconstruction, firing_rate = main_lr(
                network_type=network_type,
                num_neurons=num_neurons,
                tuning_function=p if tuning_function is None else tuning_function,
                cluster=p if cluster is None else cluster,
                num_patches=p if patches is None else patches,
                perlin_input_cluster=p if perlin_input_cluster is None else perlin_input_cluster,
                rec_factor=p if rec_factor is None else rec_factor,
                c_alpha=p if c_alpha is None else c_alpha,
                img_prop=img_prop,
                presentation_time=presentation_time,
                spatial_sampling=spatial_sampling,
                use_equilibrium=use_equilibrium,
                write_to_file=True,
                load_network=load_network,
                save_plots=save_plots,
                save_prefix=save_prefix,
                verbosity=verbosity
            )

            ed = error_distance(input_stimulus, reconstruction)
            ed_file = open(curr_dir + "/error/%s_error_distance.txt" % save_prefix, "w+")
            ed_file.write(str(ed))
            ed_file.close()

            errors.append(ed)

        # #############################################################################################################
        # Write values to file
        # #############################################################################################################

        if verbosity > 0:
            print_msg("Mean Error for network type %s, %s %s, image proportion %s, "
                      "and input type %s: %s \n"
                  % (
                      network_name,
                      parameter_str,
                      p if tuning_function is not None else tuning_name,
                      img_prop,
                      input_name,
                      np.asarray(errors).mean()
                  ))
            print_msg("Error variance for network type %s, %s %s, image proportion %s, "
                      "and input type %s: %s \n"
                  % (
                      network_name,
                      parameter_str,
                      p if tuning_function is not None else tuning_name,
                      img_prop,
                      input_name,
                      np.asarray(errors).mean()
                  ))


def main():
    # ################################################################################################################
    # Initialise parameters
    # ################################################################################################################
    network_type = None
    num_neurons = int(1e4)
    tuning_function = TUNING_FUNCTION["gauss"]
    cluster = (15, 15)
    perlin_input_cluster = (4, 4)
    num_trials = 10
    patches = 3
    c_alpha = 0.7
    rec_factor = 1.
    img_prop = 1.
    presentation_time = 1000.
    spatial_sampling = False
    save_plots = True
    use_equilibrium = False
    verbosity = VERBOSITY
    load_network = False
    existing_ok = False

    # ################################################################################################################
    # Parse command line arguments
    # ################################################################################################################
    cmd_params = arg_parse(sys.argv[1:])
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

    if cmd_params.num_neurons is not None:
        num_neurons = int(cmd_params.num_neurons)

    if cmd_params.perlin is not None:
        perlin_input_cluster = (cmd_params.perlin, cmd_params.perlin)

    if cmd_params.parameter in list(PARAMETER_DICT.keys()):
        if cmd_params.parameter.lower() == "tuning":
            tuning_function = None
        elif cmd_params.parameter.lower() == "patches":
            if "patchy" not in cmd_params.network.lower():
                raise ValueError("Cannot run experiments about the number of patches a non-patchy network")
            patches = None
        elif cmd_params.parameter.lower() == "alpha":
            if "patchy" not in cmd_params.network.lower():
                raise ValueError("Cannot run experiments about the different alpha values when no patches present")
            c_alpha = None
        elif cmd_params.parameter.lower() == "cluster":
            if network_type == NETWORK_TYPE["random"]:
                raise ValueError("Cannot run experiments about the cluster size with a random network")
            cluster = None
        elif cmd_params.parameter.lower() == "weights":
            rec_factor = None

    if cmd_params.tuning is not None:
        if tuning_function is not None:
            tuning_function = TUNING_FUNCTION[cmd_params.tuning]
        else:
            raise ValueError("Cannot pass 'tuning' as experimental parameter and set tuning function")

    if cmd_params.cluster is not None:
        if cluster is not None:
            cluster = (cmd_params.cluster, cmd_params.cluster)
        else:
            raise ValueError("Cannot pass 'cluster' as experimental parameter and set cluster")

    if cmd_params.patches is not None:
        if patches is not None:
            patches = cmd_params.patches
        else:
            raise ValueError("Cannot pass 'patches' as experimental parameter and set patches")

    if cmd_params.c_alpha is not None:
        if c_alpha is not None:
            c_alpha = cmd_params.c_alpha
        else:
            raise ValueError("Cannot pass 'alpha' as experimental parameter and set c_alpha")

    if cmd_params.rec_factor is not None:
        if rec_factor is not None:
            rec_factor = cmd_params.rec_factor
        else:
            raise ValueError("Cannot pass 'weights' as experimental parameter and set feedforward weight factor")

    if cmd_params.load_network:
        load_network = True

    if cmd_params.num_trials is not None:
        num_trials = cmd_params.num_trials

    if cmd_params.img_prop is not None:
        img_prop = float(cmd_params.img_prop)

    if cmd_params.verbosity is not None:
        verbosity = cmd_params.verbosity

    if cmd_params.equilibrium:
        use_equilibrium = True

    if cmd_params.existing_ok:
        existing_ok = True

    print("Start experiments for network %s given the Perlin resolution is %s."
          " The parameter %s is changed."
          " The number of trials is %s."
          " For the reconstruction methods, the equilibrium state of the network is%s used"
          " and sampling rate is %s with%s spatial correlation"
          % (
              cmd_params.network,
              perlin_input_cluster[0],
              cmd_params.parameter,
              num_trials,
              "" if use_equilibrium else " not",
              img_prop,
              "" if spatial_sampling else "out"
          ))

    if cmd_params.presentation_time is not None:
        presentation_time = cmd_params.presentation_time

    # ################################################################################################################
    # Run experiment
    # ################################################################################################################
    experiment(
        network_type=network_type,
        num_neurons=num_neurons,
        tuning_function=tuning_function,
        cluster=cluster,
        perlin_input_cluster=perlin_input_cluster,
        patches=patches,
        rec_factor=rec_factor,
        img_prop=img_prop,
        c_alpha=c_alpha,
        presentation_time=presentation_time,
        spatial_sampling=spatial_sampling,
        use_equilibrium=use_equilibrium,
        save_plots=save_plots,
        load_network=load_network,
        existing_ok=existing_ok,
        num_trials=num_trials,
        verbosity=verbosity
    )


if __name__ == '__main__':
    main()

