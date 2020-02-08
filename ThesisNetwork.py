#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.stimulusReconstruction import stimulus_reconstruction
from modules.networkConstruction import *
from modules.createStimulus import *
from modules.networkAnalysis import *

import numpy as np
import matplotlib.pyplot as plt

import nest


VERBOSITY = 3
nest.set_verbosity("M_ERROR")


def create_network(input_stimulus, cap_s=1., receptor_connect_strength=1., use_patchy=True, use_stimulus_local=True):
    # #################################################################################################################
    # Define values
    # #################################################################################################################
    # Not necessary to divide by 10, as sparsity is obtained by tuning preference ?
    num_sensory = input_stimulus.size // 10
    num_stimulus_discr = 4
    rf_size = (input_stimulus.shape[0] / 3., input_stimulus.shape[1] / 3.)
    ignore_weights_adj = False
    patchy_connect_dict = {"rule": "fixed_indegree", "indegree": 25}
    rf_connect_dict = {"rule": "pairwise_bernoulli", "p": 0.7}
    local_connect_dict = {"rule": "pairwise_bernoulli", "p": 0.7}
    plot_rf_relation = False if VERBOSITY < 4 else True
    p_loc = 0.5
    # #################################################################################################################
    # Create nodes and orientation map
    # #################################################################################################################
    if VERBOSITY > 0:
        print("\n#####################\tCreate receptor layer")
    receptor_layer = create_input_current_generator(input_stimulus, organise_on_grid=True)

    if VERBOSITY > 0:
        print("\n#####################\tCreate sensory layer")
    torus_layer, spike_detect, _ = create_torus_layer_uniform(num_sensory)
    torus_layer_nodes = nest.GetNodes(torus_layer)[0]
    receptor_layer_nodes = nest.GetNodes(receptor_layer)[0]

    # Create stimulus tuning map
    if VERBOSITY > 0:
        print("\n#####################\tCreate stimulus tuning map")
    tuning_to_neuron_map, neuron_to_tuning_map, tuning_weight_vector = create_stimulus_tuning_map(
        torus_layer,
        num_stimulus_discr=num_stimulus_discr
    )

    if VERBOSITY > 0:
        print("\n#####################\tCreate central points for receptive fields")
    sens_node_positions = tp.GetPosition(torus_layer_nodes)
    rf_center_map = [
        ((x / (R_MAX / 2.)) * input_stimulus.shape[1] / 2., (y / (R_MAX / 2.)) * input_stimulus.shape[0] / 2.)
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
        synaptic_strength=receptor_connect_strength
    )

    if VERBOSITY > 0:
        print("\n#####################\tCreate local connections")
    if not use_patchy:
        if use_stimulus_local:
            local_connect_dict={"rule": "pairwise_bernoulli", "p": 1.}
        else:
            p_loc = 1.

    # Create local connections
    if use_stimulus_local:
        create_stimulus_based_local_connections(
            torus_layer,
            neuron_to_tuning_map,
            tuning_to_neuron_map,
            connect_dict=local_connect_dict
        )
    else:
        create_local_circular_connections(torus_layer, p_loc=p_loc)

    # Create long-range patchy connections
    if use_patchy:
        if VERBOSITY > 0:
            print("\n#####################\tCreate long-range patchy connections")
        create_stimulus_based_patches_random(
            torus_layer,
            neuron_to_tuning_map,
            tuning_to_neuron_map,
            connect_dict=patchy_connect_dict
        )

    # Create sensory-to-sensory matrix
    if VERBOSITY > 0:
        print("\n#####################\tCreate adjacency matrix for sensory-to-sensory connections")
    adj_sens_sens_mat = create_adjacency_matrix(torus_layer_nodes, torus_layer_nodes)

    # Set sensory-to-sensory weights
    if VERBOSITY > 0:
        print("\n#####################\tSet synaptic weights for sensory to sensory neurons")
    set_synaptic_strenght(torus_layer_nodes, adj_sens_sens_mat, cap_s=cap_s)

    return receptor_layer, torus_layer, adj_rec_sens_mat, adj_sens_sens_mat, tuning_weight_vector, spike_detect


def main_lr(use_patchy=True):
    # load input stimulus
    # input_stimulus = image_with_spartial_correlation(size_img=(50, 50))
    input_stimulus = load_image("nfl-sunflower50.jpg")
    if VERBOSITY > 2:
        plt.imshow(input_stimulus, cmap='gray')
        plt.show()

    # #################################################################################################################
    # Define values
    # #################################################################################################################
    num_receptor = input_stimulus.size
    simulation_time = 250.
    use_mask = False
    plot_only_eigenspectrum = True
    cap_s = 1.
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
        use_patchy=use_patchy
    )

    torus_layer_nodes = nest.GetNodes(torus_layer)[0]
    if plot_only_eigenspectrum:
        _, _ = eigenvalue_analysis(
            adj_sens_sens_mat,
            plot=True,
            save_plot=True,
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
    if VERBOSITY > 1:
        plt.plot(time_s, spikes_s, "k,")
        plt.show()

    firing_rates = get_firing_rates(spikes_s, torus_layer_nodes, simulation_time)

    if VERBOSITY > 0:
        average_firing_rate = np.mean(firing_rates)
        print("\n#####################\tAverage firing rate: %s \n" % average_firing_rate)

    # #################################################################################################################
    # Reconstruct stimulus
    # #################################################################################################################
    mask = np.ones(firing_rates.shape, dtype='bool')
    if use_mask:
        mask = firing_rates > 0

    # Reconstruct input stimulus
    if VERBOSITY > 0:
        print("\n#####################\tReconstruct stimulus")
    reconstruction = stimulus_reconstruction(
        firing_rates[mask],
        cap_s / float(adj_sens_sens_mat.sum()),
        receptor_connect_strength,
        adj_rec_sens_mat[:, mask],
        adj_sens_sens_mat[mask, mask],
        stimulus_size=num_receptor,
        tuning_weight_vector=tuning_weight_vector[mask]
    )

    if VERBOSITY > 1:
        _, fig = plt.subplots(1, 2)
        fig[0].imshow(reconstruction, cmap='gray')
        fig[1].imshow(input_stimulus, cmap='gray')
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
    # np.random.seed(0)
    main_lr()