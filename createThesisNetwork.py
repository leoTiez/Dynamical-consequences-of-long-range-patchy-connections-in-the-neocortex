#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.networkConstruction import *
from modules.createStimulus import *
from modules.networkAnalysis import *

import nest

nest.set_verbosity("M_ERROR")


def create_network(
        input_stimulus,
        cap_s=1.,
        receptor_connect_strength=1.,
        ignore_weights_adj=False,
        use_patchy=True,
        use_stimulus_local=True,
        verbosity=0,
):
    # #################################################################################################################
    # Define values
    # #################################################################################################################
    # Not necessary to divide by 10, as sparsity is obtained by tuning preference ?
    num_sensory = input_stimulus.size  # // 10
    num_stimulus_discr = 4
    num_patches = 3
    rf_size = (input_stimulus.shape[0] / 3., input_stimulus.shape[1] / 3.)
    patchy_connect_dict = {"rule": "pairwise_bernoulli", "p": 0.2}
    rf_connect_dict = {"rule": "pairwise_bernoulli", "p": 0.4}
    local_connect_dict = {"rule": "pairwise_bernoulli", "p": 0.7}
    p_loc = 0.5
    pot_threshold = 1e3
    capacitance = 1e12
    layer_size = 3.

    plot_rf_relation = False if verbosity < 4 else True
    plot_tuning_map = False if verbosity < 4 else True
    plot_local_connections = False if verbosity < 4 else True
    plot_patchy_connections = False if verbosity < 4 else True
    save_plots = True

    # #################################################################################################################
    # Create nodes and orientation map
    # #################################################################################################################
    if verbosity > 0:
        print("\n#####################\tCreate receptor layer")
    receptor_layer = create_input_current_generator(input_stimulus, organise_on_grid=True)

    if verbosity > 0:
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
    if verbosity > 0:
        print("\n#####################\tCreate stimulus tuning map")
    tuning_to_neuron_map, neuron_to_tuning_map, tuning_weight_vector, color_map = create_perlin_stimulus_map(
        torus_layer,
        num_stimulus_discr=num_stimulus_discr,
        plot=plot_tuning_map,
        resolution=(20, 20),
        save_plot=save_plots
    )

    if verbosity > 0:
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
    if verbosity > 0:
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

    if verbosity > 0:
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
        if verbosity > 0:
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

    if verbosity > 3:
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
    if verbosity > 0:
        print("\n#####################\tCreate adjacency matrix for sensory-to-sensory connections")
    adj_sens_sens_mat = create_adjacency_matrix(torus_layer_nodes, torus_layer_nodes)

    # Set sensory-to-sensory weights
    if verbosity > 0:
        print("\n#####################\tSet synaptic weights for sensory to sensory neurons")
    set_synaptic_strength(torus_layer_nodes, adj_sens_sens_mat, cap_s=cap_s)

    return receptor_layer, torus_layer, adj_rec_sens_mat, adj_sens_sens_mat, tuning_weight_vector, spike_detect

