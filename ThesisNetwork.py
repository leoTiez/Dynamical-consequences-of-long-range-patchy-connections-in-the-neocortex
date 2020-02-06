#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.stimulusReconstruction import stimulus_reconstruction
from modules.networkConstruction import *
from modules.createStimulus import *
from modules.networkAnalysis import *

import numpy as np
import matplotlib.pyplot as plt

import nest


VERBOSITY = 2
nest.set_verbosity("M_ERROR")


def main_lr(use_patchy=True):
    # load input stimulus
    # input_stimulus = image_with_spartial_correlation(size_img=(50, 50))
    input_stimulus = load_image("nfl-planet.PNG")
    if VERBOSITY > 2:
        plt.imshow(input_stimulus, cmap='gray')
        plt.show()

    # #################################################################################################################
    # Define values
    # #################################################################################################################
    num_receptor = input_stimulus.size
    # Not necessary to divide by 10, as sparsity is obtained by tuning preference ?
    num_sensory = input_stimulus.size // 5
    num_stimulus_discr = 8
    simulation_time = 250.
    cap_s = 1.
    receptor_connect_strength = 1.
    rf_size = (3, 3)
    ignore_weights_adj = True
    patchy_connect_dict = {"rule": "fixed_indegree", "indegree": 25}
    rf_connect_dict = {"rule": "fixed_indegree", "indegree": 5}
    use_mask = True
    p_loc = 0.5

    # #################################################################################################################
    # Create nodes and orientation map
    # #################################################################################################################
    receptor_layer = create_input_current_generator(input_stimulus, organise_on_grid=True)
    torus_layer, spike_detect, _ = create_torus_layer_uniform(num_sensory)
    torus_layer_nodes = nest.GetNodes(torus_layer)[0]
    receptor_layer_nodes = nest.GetNodes(receptor_layer)[0]
    # Create stimulus tuning map
    tuning_to_neuron_map, neuron_to_tuning_map = create_stimulus_tuning_map(
        torus_layer,
        num_stimulus_discr=num_stimulus_discr
    )

    # Create map for receptive field centers for neurons
    x_rf = np.random.choice(input_stimulus.shape[1], size=num_sensory)
    y_rf = np.random.choice(input_stimulus.shape[0], size=num_sensory)
    rf_center_map = zip(x_rf.astype('float'), y_rf.astype('float'))

    # #################################################################################################################
    # Create Connections
    # #################################################################################################################
    # Create connections to receptive field
    adj_rec_sens_mat = create_connections_rf(
        receptor_layer,
        torus_layer,
        rf_center_map,
        neuron_to_tuning_map,
        connect_dict=rf_connect_dict,
        rf_size=rf_size,
        ignore_weights=ignore_weights_adj
    )
    if not use_patchy:
        p_loc = 1.
    # Create local connections
    create_local_circular_connections(torus_layer, p_loc=p_loc)
    # Create long-range patchy connections
    if use_patchy:
        create_stimulus_based_patches_random(
            torus_layer,
            neuron_to_tuning_map,
            tuning_to_neuron_map,
            connect_dict=patchy_connect_dict
        )
    # Create sensory-to-sensory matrix
    adj_sens_sens_mat = create_adjacency_matrix(torus_layer_nodes, torus_layer_nodes)
    # Set sensory-to-sensory weights
    set_synaptic_strenght(torus_layer_nodes, adj_sens_sens_mat, cap_s=cap_s)

    # #################################################################################################################
    # Simulate and retrieve resutls
    nest.Simulate(simulation_time)
    # #################################################################################################################

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
    reconstruction = stimulus_reconstruction(
        firing_rates[mask],
        cap_s / float(adj_sens_sens_mat.sum()),
        receptor_connect_strength,
        adj_rec_sens_mat[:, mask],
        adj_sens_sens_mat[mask, mask],
        stimulus_size=num_receptor
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