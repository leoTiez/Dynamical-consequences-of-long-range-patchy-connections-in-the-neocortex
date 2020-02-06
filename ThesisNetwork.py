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


def main_lr():
    # load input stimulus
    input_stimulus = image_with_spartial_correlation(size_img=(50, 50))
    if VERBOSITY > 2:
        plt.imshow(input_stimulus, cmap='gray')
        plt.show()

    # Define values
    num_receptor = input_stimulus.size
    # Not necessary to divide by 10, as sparsity is obtained by tuning preference ?
    num_sensory = input_stimulus.size #// 10
    num_stimulus_discr = 4
    simulation_time = 250.
    cap_s = 1.
    receptor_connect_strength = 1.
    rf_size = (5, 5)
    ignore_weights_adj = True
    patchy_p = 0.7

    # Create nodes
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
        rf_size=rf_size,
        ignore_weights=ignore_weights_adj
    )
    # Create local connections
    create_local_circular_connections(torus_layer)
    # Create long-range patchy connections
    create_stimulus_based_patches_random(torus_layer, neuron_to_tuning_map, tuning_to_neuron_map, p_p=patchy_p)
    # Create sensory-to-sensory matrix
    adj_sens_sens_mat = create_adjacency_matrix(torus_layer_nodes, torus_layer_nodes)
    # Set sensory-to-sensory weights
    set_synaptic_strenght(torus_layer_nodes, adj_sens_sens_mat, cap_s=cap_s)

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


if __name__ == '__main__':
    np.random.seed(0)
    main_lr()