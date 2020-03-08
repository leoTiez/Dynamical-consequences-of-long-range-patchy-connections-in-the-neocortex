#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.networkConstruction import *
from modules.createStimulus import *
from modules.networkAnalysis import *

from scipy.spatial import KDTree
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
        ff_weight=1.,
        network_type="local_radial_lr_patchy",
        verbosity=0,
        weights_connection_dependent=False,
        sens_adj_mat_needed=False
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
    num_sensory = int(1e4)
    ratio_inh_neurons = 5
    num_stimulus_discr = 4
    num_patches = 3
    p_loc = 0.5
    p_rf = 0.3
    p_lr = 0.2
    p_random = 0.001
    inh_weight = -1.
    rf_size = (input_stimulus.shape[0] // 4, input_stimulus.shape[1] // 4)
    patchy_connect_dict = {"rule": "pairwise_bernoulli", "p": p_lr}
    rf_connect_dict = {"rule": "pairwise_bernoulli", "p": p_rf}
    local_connect_dict = {"rule": "pairwise_bernoulli", "p": p_loc}
    pot_threshold = -55.
    pot_reset = -70.
    capacitance = 80.
    layer_size = 8.
    r_loc = 0.5
    spacing_perlin = 0.01
    resolution_perlin = (20, 20)

    plot_rf_relation = False if verbosity < 4 else True
    plot_tuning_map = False if verbosity < 4 else True
    plot_local_connections = False if verbosity < 4 else True
    plot_patchy_connections = False if verbosity < 4 else True
    save_plots = False

    # #################################################################################################################
    # Create nodes and orientation map
    # #################################################################################################################
    if verbosity > 0:
        print("\n#####################\tCreate sensory layer")
    torus_layer, spike_detect, _ = create_torus_layer_uniform(
        num_sensory,
        threshold_pot=pot_threshold,
        capacitance=capacitance,
        rest_pot=pot_reset,
        size_layer=layer_size
    )
    torus_layer_nodes = nest.GetNodes(torus_layer, properties={"element_type": "neuron"})[0]
    torus_layer_nodes_pos = tp.GetPosition(torus_layer_nodes)
    torus_layer_tree = KDTree(torus_layer_nodes_pos)
    torus_inh_nodes = np.random.choice(np.asarray(torus_layer_nodes), size=num_sensory // ratio_inh_neurons).tolist()

    # Create stimulus tuning map
    if verbosity > 0:
        print("\n#####################\tCreate stimulus tuning map")

    if network_value == 1:
        tuning_to_neuron_map, neuron_to_tuning_map, tuning_weight_vector, color_map = create_random_stimulus_map(
            torus_layer,
            torus_inh_nodes,
            num_stimulus_discr=num_stimulus_discr,
            plot=plot_tuning_map,
            save_plot=save_plots
        )
    else:
        tuning_to_neuron_map, neuron_to_tuning_map, tuning_weight_vector, color_map = create_perlin_stimulus_map(
            torus_layer,
            torus_inh_nodes,
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
        (
            (x + (layer_size / 2.)) / float(layer_size) * input_stimulus.shape[1],
            (y + (layer_size / 2.)) / float(layer_size) * input_stimulus.shape[0]
        )
        for (x, y) in sens_node_positions
    ]

    # #################################################################################################################
    # Create Connections
    # #################################################################################################################
    # Create connections to receptive field
    if verbosity > 0:
        print("\n#####################\tCreate connections between receptors and sensory neurons")
    adj_rec_sens_mat = create_connections_rf(
        input_stimulus,
        torus_layer,
        rf_center_map,
        neuron_to_tuning_map,
        torus_inh_nodes,
        synaptic_strength=ff_weight,
        connect_dict=rf_connect_dict,
        rf_size=rf_size,
        plot_src_target=plot_rf_relation,
        retina_size=input_stimulus.shape,
        save_plot=save_plots
    )

    if network_value == 1:
        if verbosity > 0:
            print("\n#####################\tCreate random connections")
        create_random_connections(
            torus_layer,
            torus_inh_nodes,
            inh_weight=inh_weight,
            prob=p_random,
            cap_s=cap_s,
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
                torus_layer_tree,
                neuron_to_tuning_map,
                tuning_to_neuron_map,
                torus_inh_nodes,
                inh_weight=inh_weight,
                cap_s=cap_s,
                connect_dict=local_connect_dict,
                r_loc=r_loc,
                plot=plot_local_connections,
                color_mask=color_map,
                save_plot=save_plots
            )
        elif network_value in [2, 4, 5]:
            create_local_circular_connections(
                torus_layer,
                torus_inh_nodes,
                inh_weight=inh_weight,
                p_loc=p_loc,
                r_loc=r_loc,
                cap_s=cap_s,
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
                torus_inh_nodes,
                torus_layer_tree,
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
                torus_inh_nodes,
                r_loc=r_loc,
                p_loc=p_loc,
                cap_s=cap_s,
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

    # Create sensory-to-sensory matrix
    adj_sens_sens_mat = None
    if sens_adj_mat_needed:
        if verbosity > 0:
            print("\n#####################\tCreate adjacency matrix for sensory-to-sensory connections")
        adj_sens_sens_mat = create_adjacency_matrix(torus_layer_nodes, torus_layer_nodes)

    # Set sensory-to-sensory weights
    if weights_connection_dependent:
        if verbosity > 0:
            print("\n#####################\tSet synaptic weights for sensory to sensory neurons")
        set_synaptic_strength(torus_layer_nodes, adj_sens_sens_mat, cap_s=cap_s, divide_by_num_connect=True)

    return (
        torus_layer_nodes,
        adj_rec_sens_mat,
        adj_sens_sens_mat,
        tuning_weight_vector,
        spike_detect,
        color_map
    )

