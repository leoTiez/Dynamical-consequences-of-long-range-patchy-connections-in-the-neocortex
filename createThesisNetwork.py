#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.networkConstruction import *
from modules.createStimulus import *
from modules.networkAnalysis import *

import nest

nest.set_verbosity("M_ERROR")

NETWORK_TYPE = {
    "random": 1,
    "local_radial": 2,
    "local_stimulus": 3,
    "local_radial_lr_random": 4,
    "local_radial_lr_patchy": 5,
    "local_stimulus_lr_patchy": 6
}


def create_network(
        input_stimulus,
        cap_s=1.,
        receptor_connect_strength=1.,
        network_type="local_radial_lr_patchy",
        ignore_weights_adj=False,
        verbosity=0,
):
    # #################################################################################################################
    # Get network type
    # #################################################################################################################
    try:
        network_value = NETWORK_TYPE[network_type.lower()]
    except KeyError:
        raise KeyError("Network type %s is not supported" % network_type.lower())

    # #################################################################################################################
    # Define values
    # #################################################################################################################
    # Not necessary to divide by 10, as sparsity is obtained by tuning preference ?
    num_sensory = input_stimulus.size
    num_stimulus_discr = 4
    num_patches = 3
    p_loc = 0.5
    p_rf = 0.3
    p_lr = 0.2
    p_random = 0.05
    p_inh = 0.1
    rf_size = (input_stimulus.shape[0] / 4., input_stimulus.shape[1] / 4.)
    patchy_connect_dict = {"rule": "pairwise_bernoulli", "p": p_lr}
    rf_connect_dict = {"rule": "pairwise_bernoulli", "p": p_rf}
    local_connect_dict = {"rule": "pairwise_bernoulli", "p": p_loc}
    pot_threshold = -55.
    pot_reset = -70.
    capacitance = 80.
    layer_size = 3.
    r_loc = 0.3
    spacing_perlin = 0.01
    resolution_perlin = (10, 10)

    plot_rf_relation = False if verbosity < 4 else True
    plot_tuning_map = False if verbosity < 4 else True
    plot_local_connections = False if verbosity < 4 else True
    plot_patchy_connections = False if verbosity < 4 else True
    save_plots = False

    # #################################################################################################################
    # Create nodes and orientation map
    # #################################################################################################################
    if verbosity > 0:
        print("\n#####################\tCreate receptor layer")
    receptor_layer = create_input_current_generator(input_stimulus, organise_on_grid=True)

    if verbosity > 0:
        print("\n#####################\tCreate sensory layer")
    (torus_layer, torus_layer_inh), spike_detect, _ = create_torus_layer_with_inh(
        num_sensory,
        threshold_pot=pot_threshold,
        capacitance=capacitance,
        rest_pot=pot_reset,
        size_layer=layer_size
    )
    torus_layer_nodes = nest.GetNodes(torus_layer)[0]
    receptor_layer_nodes = nest.GetNodes(receptor_layer)[0]

    # Create stimulus tuning map
    if verbosity > 0:
        print("\n#####################\tCreate stimulus tuning map")

    if network_value == 1:
        tuning_to_neuron_map, neuron_to_tuning_map, tuning_weight_vector, color_map = create_random_stimulus_map(
            torus_layer,
            num_stimulus_discr=num_stimulus_discr,
            plot=plot_tuning_map,
            save_plot=save_plots
        )
    else:
        tuning_to_neuron_map, neuron_to_tuning_map, tuning_weight_vector, color_map = create_perlin_stimulus_map(
            torus_layer,
            num_stimulus_discr=num_stimulus_discr,
            plot=plot_tuning_map,
            spacing=spacing_perlin,
            resolution=resolution_perlin,
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

    if network_value == 1:
        if verbosity > 0:
            print("\n#####################\tCreate random connections")
        create_random_connections(
            torus_layer,
            prob=p_random,
            plot=plot_local_connections,
            save_plot=save_plots,
            color_mask=color_map
        )

    else:
        if verbosity > 0:
            print("\n#####################\tCreate local connections")
        # Create local connections
        if network_value in [3, 6]:
            create_stimulus_based_local_connections(
                torus_layer,
                neuron_to_tuning_map,
                tuning_to_neuron_map,
                connect_dict=local_connect_dict,
                r_loc=r_loc,
                plot=plot_local_connections,
                color_mask=color_map,
                save_plot=save_plots
            )
        elif network_value in [2, 4, 5]:
            create_local_circular_connections(
                torus_layer,
                p_loc=p_loc,
                r_loc=r_loc,
                plot=plot_local_connections,
                color_mask=color_map,
                save_plot=save_plots
            )

        # Create long-range patchy connections
        if network_value in [5, 6]:
            if verbosity > 0:
                print("\n#####################\tCreate long-range patchy stimulus dependent connections")
            create_stimulus_based_patches_random(
                torus_layer,
                neuron_to_tuning_map,
                tuning_to_neuron_map,
                r_loc=r_loc,
                connect_dict=patchy_connect_dict,
                num_patches=num_patches,
                plot=plot_patchy_connections,
                save_plot=save_plots,
                color_mask=color_map
            )

        if network_value == 4:
            if verbosity > 0:
                print("\n#####################\tCreate long-range patchy random connections")
            create_random_patches(
                torus_layer,
                r_loc=r_loc,
                p_loc=p_loc,
                p_p=patchy_connect_dict["p"],
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

    if verbosity > 0:
        print("\n#####################\tCreate inhibitory connections")
    create_inh_exc_connections(layer_exc=torus_layer, layer_inh=torus_layer_inh, prob=p_inh)
    create_connections_rf(
        receptor_layer,
        torus_layer_inh,
        rf_center_map,
        neuron_to_tuning_map,
        no_tuning=True,
        connect_dict=rf_connect_dict,
        rf_size=rf_size,
        ignore_weights=ignore_weights_adj,
        plot_src_target=plot_rf_relation,
        retina_size=input_stimulus.shape,
        synaptic_strength=receptor_connect_strength,
        save_plot=save_plots
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

