#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.stimulusReconstruction import stimulus_reconstruction, fourier_trans, direct_stimulus_reconstruction
from modules.networkConstruction import *
from modules.createStimulus import *
from modules.networkAnalysis import *

import numpy as np
import matplotlib.pyplot as plt

import nest


VERBOSITY = 4
nest.set_verbosity("M_ERROR")


def create_network(
        input_stimulus,
        cap_s=1.,
        receptor_connect_strength=1.,
        ignore_weights_adj=False,
        use_patchy=True,
        use_stimulus_local=True
):
    # #################################################################################################################
    # Define values
    # #################################################################################################################
    # Not necessary to divide by 10, as sparsity is obtained by tuning preference ?
    num_sensory = input_stimulus.size // 10
    num_stimulus_discr = 4
    num_patches = 3
    rf_size = (input_stimulus.shape[0] / 3., input_stimulus.shape[1] / 3.)
    patchy_connect_dict = {"rule": "pairwise_bernoulli", "p": 0.4}
    rf_connect_dict = {"rule": "pairwise_bernoulli", "p": 0.1}
    local_connect_dict = {"rule": "pairwise_bernoulli", "p": 0.6}
    p_loc = 0.5
    pot_threshold = 1e3
    capacitance = 1e12
    layer_size = 3.
    stimulus_per_row = 2

    plot_rf_relation = False if VERBOSITY < 4 else True
    plot_tuning_map = False if VERBOSITY < 4 else True
    plot_local_connections = False if VERBOSITY < 4 else True
    plot_patchy_connections = False if VERBOSITY < 4 else True
    save_plots = True

    # #################################################################################################################
    # Create nodes and orientation map
    # #################################################################################################################
    if VERBOSITY > 0:
        print("\n#####################\tCreate receptor layer")
    receptor_layer = create_input_current_generator(input_stimulus, organise_on_grid=True)

    if VERBOSITY > 0:
        print("\n#####################\tCreate sensory layer")
    torus_layer, spike_detect, _ = create_torus_layer_uniform(
        num_sensory,
        threshold_pot=pot_threshold,
        capacitance=capacitance,
        size_layer=layer_size
    )
    torus_layer_nodes = nest.GetNodes(torus_layer)[0]
    receptor_layer_nodes = nest.GetNodes(receptor_layer)[0]

    # Create stimulus tuning map
    if VERBOSITY > 0:
        print("\n#####################\tCreate stimulus tuning map")
    tuning_to_neuron_map, neuron_to_tuning_map, tuning_weight_vector, color_map = create_perlin_stimulus_map(
        torus_layer,
        num_stimulus_discr=num_stimulus_discr,
        plot=plot_tuning_map,
        resolution=(20, 20),
        save_plot=save_plots
    )

    if VERBOSITY > 0:
        print("\n#####################\tCreate central points for receptive fields")
    sens_node_positions = tp.GetPosition(torus_layer_nodes)
    rf_center_map = [
        ((x / (layer_size / 2.)) * input_stimulus.shape[1] / 2., (y / (layer_size / 2.)) * input_stimulus.shape[0] / 2.)
        for (x, y) in sens_node_positions
    ]

    # #################################################################################################################
    # Create Connections
    # #################################################################################################################
    # Create connections to receptive field
    if VERBOSITY > 0:
        print("\n#####################\tCreate connections between receptors and sensory neurons")
    adj_rec_sens_mat = create_connections_rf(
        receptor_layer,
        torus_layer,
        rf_center_map,
        neuron_to_tuning_map,
        connect_dict=rf_connect_dict,
        rf_size=rf_size,
        ignore_weights=ignore_weights_adj,
        plot_src_target=plot_rf_relation,
        retina_size=input_stimulus.shape,
        synaptic_strength=receptor_connect_strength,
        save_plot=save_plots
    )

    if VERBOSITY > 0:
        print("\n#####################\tCreate local connections")
    if not use_patchy:
        if use_stimulus_local:
            local_connect_dict = {"rule": "pairwise_bernoulli", "p": 1.}
        else:
            p_loc = 1.

    # Create local connections
    if use_stimulus_local:
        create_stimulus_based_local_connections(
            torus_layer,
            neuron_to_tuning_map,
            tuning_to_neuron_map,
            connect_dict=local_connect_dict,
            plot=plot_local_connections,
            color_mask=color_map,
            save_plot=save_plots
        )
    else:
        create_local_circular_connections(
            torus_layer,
            p_loc=p_loc,
            plot=plot_local_connections,
            color_mask=color_map,
            save_plot=save_plots
        )

    # Create long-range patchy connections
    if use_patchy:
        if VERBOSITY > 0:
            print("\n#####################\tCreate long-range patchy connections")
        create_stimulus_based_patches_random(
            torus_layer,
            neuron_to_tuning_map,
            tuning_to_neuron_map,
            connect_dict=patchy_connect_dict,
            num_patches=num_patches,
            plot=plot_patchy_connections,
            save_plot=save_plots,
            color_mask=color_map
        )

    if VERBOSITY > 3:
        connect = nest.GetConnections([torus_layer_nodes[0]])
        targets = nest.GetStatus(connect, "target")
        sensory_targets = [t for t in targets if t in list(torus_layer_nodes)]
        plot_connections(
            [torus_layer_nodes[0]],
            sensory_targets,
            layer_size,
            save_plot=save_plots,
            plot_name="all_connections.png",
            color_mask=color_map
        )

    # Create sensory-to-sensory matrix
    if VERBOSITY > 0:
        print("\n#####################\tCreate adjacency matrix for sensory-to-sensory connections")
    adj_sens_sens_mat = create_adjacency_matrix(torus_layer_nodes, torus_layer_nodes)

    # Set sensory-to-sensory weights
    if VERBOSITY > 0:
        print("\n#####################\tSet synaptic weights for sensory to sensory neurons")
    set_synaptic_strength(torus_layer_nodes, adj_sens_sens_mat, cap_s=cap_s)

    return receptor_layer, torus_layer, adj_rec_sens_mat, adj_sens_sens_mat, tuning_weight_vector, spike_detect


def main_matrix_dynamics():
    input_stimulus = image_with_spatial_correlation(size_img=(50, 50), num_circles=5, background_noise=False)
    if VERBOSITY > 2:
        plt.imshow(input_stimulus, cmap='gray')
        plt.show()
    # #################################################################################################################
    # Define values
    # #################################################################################################################
    cap_s = 1.
    receptor_connect_strength = 1.
    plot_arrangement_rows = 10
    plot_arrangement_columns = 10

    (_, _, adj_rec_sens_mat, adj_sens_sens_mat, _, _) = create_network(
        input_stimulus,
        cap_s=cap_s,
        receptor_connect_strength=receptor_connect_strength,
        ignore_weights_adj=False,
        use_stimulus_local=False
    )

    # Sufficient to use only 255 as we don't use the neurons themselves
    input_matrix = np.full((50, 50), 255)
    sensory_activity = adj_rec_sens_mat.T.dot(input_matrix.reshape(-1))
    fig, axes = plt.subplots(plot_arrangement_rows, plot_arrangement_columns)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    for ax in axes.reshape(-1):
        ax.imshow(sensory_activity.reshape((10, 25)), cmap='gray')
        sensory_activity = adj_sens_sens_mat.T.dot(sensory_activity)

    plt.show()


def main_lr(use_patchy=True):
    # load input stimulus
    input_stimulus = image_with_spatial_correlation(size_img=(50, 50), num_circles=5, radius=15, background_noise=True)
    stimulus_fft = fourier_trans(input_stimulus)
    if VERBOSITY > 2:
        plt.imshow(input_stimulus, cmap='gray')
        plt.show()

    # #################################################################################################################
    # Define values
    # #################################################################################################################
    num_receptor = input_stimulus.size
    simulation_time = 1000.
    use_mask = False
    plot_only_eigenspectrum = False
    ignore_weights_adj = True
    cap_s = 20.     # Increased to reduce the effect of the input and to make it easier to investigate the dynamical
                    # consequences of local / lr patchy connections
    receptor_connect_strength = 1.

    (receptor_layer,
     torus_layer,
     adj_rec_sens_mat,
     adj_sens_sens_mat,
     tuning_weight_vector,
     spike_detect) = create_network(
        input_stimulus,
        cap_s=cap_s,
        receptor_connect_strength=receptor_connect_strength,
        use_patchy=use_patchy,
        ignore_weights_adj=ignore_weights_adj,
        use_stimulus_local=False
    )

    torus_layer_nodes = nest.GetNodes(torus_layer)[0]
    if plot_only_eigenspectrum:
        _, _ = eigenvalue_analysis(
            adj_sens_sens_mat,
            plot=True,
            save_plot=False,
            fig_name="thesis_network_non-zero_connections.png"
        )
        return

    # #################################################################################################################
    # Simulate and retrieve resutls
    # #################################################################################################################
    if VERBOSITY > 0:
        print("\n#####################\tSimulate")
    nest.Simulate(simulation_time)

    # Get network response in spikes
    data_sp = nest.GetStatus(spike_detect, keys="events")[0]
    spikes_s = data_sp["senders"]
    time_s = data_sp["times"]
    if VERBOSITY > 2:
        plt.plot(time_s, spikes_s, "k,")
        plt.show()

    firing_rates = get_firing_rates(spikes_s, torus_layer_nodes, simulation_time)

    if VERBOSITY > 0:
        average_firing_rate = np.mean(firing_rates)
        print("\n#####################\tAverage firing rate: %s \n" % average_firing_rate)

    if VERBOSITY > 2:
        plt.imshow(firing_rates.reshape(25, 10))
        plt.show()

    # #################################################################################################################
    # Reconstruct stimulus
    # #################################################################################################################
    mask = np.ones(firing_rates.shape, dtype='bool')
    if use_mask:
        mask = firing_rates > 0

    # Reconstruct input stimulus
    if VERBOSITY > 0:
        print("\n#####################\tReconstruct stimulus")

    # reconstruction = stimulus_reconstruction(
    #     firing_rates[mask],
    #     cap_s / float(adj_sens_sens_mat.sum()),
    #     receptor_connect_strength,
    #     adj_rec_sens_mat[:, mask],
    #     adj_sens_sens_mat[mask, mask],
    #     stimulus_size=num_receptor,
    #     tuning_weight_vector=tuning_weight_vector[mask],
    #     verbosity=True if VERBOSITY > 1 else False
    # )

    reconstruction = direct_stimulus_reconstruction(
        firing_rates[mask],
        adj_rec_sens_mat,
        tuning_weight_vector
    )
    response_fft = fourier_trans(reconstruction)

    if VERBOSITY > 1:
        from matplotlib.colors import LogNorm
        _, fig = plt.subplots(1, 2)
        fig[0].imshow(np.abs(stimulus_fft), norm=LogNorm(vmin=5))
        fig[1].imshow(np.abs(response_fft), norm=LogNorm(vmin=5))
        _, fig_2 = plt.subplots(1, 2)
        fig_2[0].imshow(reconstruction, cmap='gray')
        fig_2[1].imshow(input_stimulus, cmap='gray')
        plt.show()

    return input_stimulus, reconstruction, firing_rates


def main_mi():
    # Define parameters outside  the loop
    use_patchy_flags = [True, False]
    num_trials = 5
    for use_patchy in use_patchy_flags:
        input_stimuli = []
        reconstructed_stimuli = []
        for _ in range(num_trials):
            input_stimulus, reconstruction, _ = main_lr(use_patchy)
            input_stimuli.append(input_stimulus.reshape(-1))
            reconstructed_stimuli.append(reconstruction.reshape(-1))

        mutual_information = mutual_information_hist(input_stimuli, reconstructed_stimuli)
        connection_type = "patchy connections" if use_patchy else "local connections"
        print("\n#####################\tMutual Information MI for %s: %s \n" % (connection_type, mutual_information))


if __name__ == '__main__':
    np.random.seed(0)
    # main_mi()
    main_lr()
    # main_matrix_dynamics()


