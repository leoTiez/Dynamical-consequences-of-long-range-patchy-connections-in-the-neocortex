#!/usr/bin/python
from modules.thesisUtils import *
from modules.networkAnalysis import *
from modules.thesisConstants import *

import warnings
from pathlib import Path
import numpy as np
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import nest.topology as tp
import nest


def _create_location_based_patches(
        layer,
        r_loc=0.5,
        p_loc=0.7,
        size_boxes=0.5,
        num_patches=3,
        num_shared_patches=6,
        num_patches_replaced=3,
        is_partially_overlapping=False,
        p_p=None
):
    """
    Function to establish patchy connections for neurons that have a location based relationship, such that
    they are in the same sublayer / box
    :param layer: Layer in which the connections should be established
    :param r_loc: Radius of local connections
    :param p_loc: Probability of local connections
    :param size_boxes: Size of a sublayer in which neurons share patches. The R_MAX / size_boxes
    should be an integer value
    :param num_patches: Number of patches per neuron
    :param num_shared_patches: Number of patches per box that are shared between the neurons
    :param num_patches_replaced: Number of patches that are replaced in x-direction (for partially overlapping patches)
    :param is_partially_overlapping: Flag for partially overlapping patches
    :return: Sublayer at size_boxes/2 for debugging purposes (plotting)
    """
    layer_size = nest.GetStatus(layer, "topology")[0]["extent"][0]
    # Calculate parameters for the patches
    r_p = r_loc / 2.
    min_distance = r_loc + r_p
    max_distance = layer_size / 2. - r_loc
    if p_p is None:
        p_p = get_lr_connection_probability_patches(r_loc, p_loc, r_p, num_patches=num_patches, layer_size=layer_size)

    # Create sublayer boxes that share same patches
    sublayer_anchors, box_mask_dict = create_distinct_sublayer_boxes(size_boxes, size_layer=layer_size)

    # To make sure to be lower than any possible anchor coordinate
    last_y_anchor = -layer_size - 1
    debug_sub_layer = None
    for anchor in sublayer_anchors:
        sub_layer = tp.SelectNodesByMask(layer, anchor, mask_obj=tp.CreateMask("rectangular", specs=box_mask_dict))

        if np.all(np.asarray(anchor) == size_boxes/2.):
            debug_sub_layer = sub_layer

        # Create pool of patches
        # Calculate radial distance and the respective coordinates for patches
        if not is_partially_overlapping or anchor[1] > last_y_anchor:
            radial_distance = np.random.uniform(min_distance, max_distance, size=num_shared_patches).tolist()
            radial_angle = np.random.uniform(0., 359., size=num_shared_patches).tolist()
        else:
            replaced_indices = np.random.choice(num_shared_patches, size=num_patches_replaced, replace=False)
            np.asarray(radial_distance)[replaced_indices] = np.random.uniform(
                min_distance,
                max_distance,
                size=num_patches_replaced
            ).tolist()
            np.asarray(radial_angle)[replaced_indices] = np.random.uniform(0., 359., size=num_patches_replaced).tolist()

        last_y_anchor = anchor[1]
        # Calculate anchors of patches
        mask_specs = {"radius": r_p}
        patches_anchors = [
            (np.asarray(anchor) + np.asarray(to_coordinates(angle, distance))).tolist()
            for angle, distance in zip(radial_angle, radial_distance)
        ]

        patchy = []
        for neuron_anchor in patches_anchors:
            patchy.append(
                tp.SelectNodesByMask(
                    layer,
                    neuron_anchor,
                    mask_obj=tp.CreateMask("circular", specs=mask_specs)
                )
            )

        # Iterate through all neurons, as patches are chosen for each neuron independently
        for neuron in sub_layer:
            neuron_patches_list = np.asarray(patchy)[
                np.random.choice(len(patches_anchors), size=num_patches, replace=False)
            ]
            neuron_patches = []
            for p in neuron_patches_list:
                neuron_patches.extend(p)

            # Define connection
            connect_dict = {
                "rule": "pairwise_bernoulli",
                "p": p_p
            }
            nest.Connect([neuron], neuron_patches, connect_dict)

    # Return last sublayer for debugging
    return debug_sub_layer


def get_local_connectivity(
        r_loc,
        p_loc,
        layer_size=R_MAX
):
    """
    Calculate local connectivity
    :param r_loc: radius of local connections
    :param p_loc: probability of establishing local connections
    :param layer_size: Size of the layer
    :return: local connectivity
    """
    inner_area = np.pi * r_loc**2
    c_loc = p_loc * inner_area / float(layer_size)**2
    return c_loc, inner_area


def get_lr_connection_probability_patches(
        r_loc,
        p_loc,
        r_p,
        num_patches=3,
        layer_size=R_MAX
):
    """
    Calculate the connection probability of long range patchy connections
    :param r_loc: radius of local connections
    :param p_loc: probability of establishing local connections
    :param r_p: patchy radius
    :param num_patches: Total number of patches
    :param layer_size: Size of the layer
    :return: long range patchy connection probability
    """
    c_loc, _ = get_local_connectivity(r_loc, p_loc, layer_size=layer_size)
    patchy_area = np.pi * r_p**2
    c_lr = GLOBAL_CONNECTIVITY - c_loc

    return np.maximum((c_lr * float(layer_size)**2) / (num_patches * patchy_area), 0.0)


def get_lr_connection_probability_np(
        r_loc,
        p_loc,
        layer_size=R_MAX
):
    """
    Calculate long range connectivity probability according to Voges Paper
    :param r_loc: local radius
    :param p_loc: local connectivity probability
    :param layer_size: size of the layer
    :return: long range connectivity probability
    """

    full_area = np.pi * R_MAX**2
    # Calculate local connectivity
    c_loc, inner_area = get_local_connectivity(r_loc, p_loc, layer_size=layer_size)
    # Calculate long range connectivity
    c_lr = GLOBAL_CONNECTIVITY - c_loc

    return np.maximum(c_lr / ((full_area - inner_area) / float(layer_size)**2), 0.0)


def create_distinct_sublayer_boxes(size_boxes, size_layer=R_MAX):
    """
    Create sublayers with distinct set of neurons
    :param size_boxes: Size of a single box
    :param size_layer: Size of the layer
    :return: The central points of the boxes (one for each box), dictionary with the mask for the sublayer (one for all)
    """
    # Create sublayer boxes that share same patches
    sublayer_anchors = [[x * size_boxes + size_boxes / 2., y * size_boxes + size_boxes / 2.]
                        for y in range(-int(size_layer / (2. * float(size_boxes))),
                                       int(size_layer / (2. * float(size_boxes))))
                        for x in range(-int(size_layer / (2. * float(size_boxes))),
                                       int(size_layer / (2. * float(size_boxes))))
                        ]
    if len(sublayer_anchors) == 0:
        sublayer_anchors = [[0., 0.]]
    box_mask_dict = {"lower_left": [-size_boxes / 2., -size_boxes / 2.],
                     "upper_right": [size_boxes / 2., size_boxes / 2.]}

    return sublayer_anchors, box_mask_dict


def create_torus_layer_uniform(
        num_neurons=3600,
        neuron_type="iaf_psc_delta",
        rest_pot=-70.,
        threshold_pot=-55.,
        time_const=20.,
        capacitance=80.,
        size_layer=R_MAX,
        p_rf=.7,
        ff_factor=1.,
        synaptic_strength=1.,
        bg_rate=300.,
        positions=None,
        to_file=False
):
    """
    Create a layer wrapped a torus to avoid boundary conditions. Neurons are placed uniformly
    :param num_neurons: Number of neurons in the layer
    :param neuron_type: Type of the neuron. So far iaf psc delta and iaf psc alpha neurons are supported
    :param rest_pot: Resting potential of the neurons
    :param threshold_pot: Threshold potential of the neurons
    :param time_const: Time constant of the neurons
    :param capacitance: Capacitance of the neurons
    :param size_layer: Size of the layer
    :param p_rf: The receptive field conenction probability. This is used to determine an adequate background activity
    :param ff_factor: The weight factor for the ff weights. This is used to determine an adequate background activity
    :param synaptic_strength: The feedforward activity. This is used to determine an adequate background activity
    :param bg_rate: Maximal spiking rate for background activity
    :param positions: If set to None, positions are generated, otherwise these positions are used. Note that the number
    of positions must match the number of neurons
    :param to_file: If set to true, the spikes are written to a file
    :return: neural layer, spike detector and mutlimeter
    """
    # Calculate positions
    if positions is None:
        positions = np.random.uniform(- size_layer / 2., size_layer / 2., size=(num_neurons, 2)).tolist()
    # Create dict for neural layer that is wrapped as torus to avoid boundary effects
    torus_dict = {
        "extent": [size_layer, size_layer],
        "positions": positions,
        "elements": neuron_type,
        "edge_wrap": True
    }

    if neuron_type == "iaf_psc_delta" or neuron_type == "iaf_psc_alpha":
        neuron_dict = {
            "V_m": rest_pot,
            "E_L": rest_pot,
            "C_m": capacitance,
            "tau_m": time_const,
            "V_th": threshold_pot,
            "V_reset": rest_pot
        }

    else:
        raise ValueError("The passed neuron type %s is not supported" % neuron_type)

    # Create layer
    torus_layer = tp.CreateLayer(torus_dict)
    sensory_nodes = nest.GetNodes(torus_layer)[0]
    nest.SetStatus(sensory_nodes, neuron_dict)
    spikedetector = nest.Create("spike_detector", params={
        "withgid": True,
        "withtime": True,
        "to_file": to_file,
        "label": "spike_detector"
    })
    multimeter = nest.Create("multimeter", params={"withtime": True, "record_from": ["V_m"]})
    nest.Connect(sensory_nodes, spikedetector)
    nest.Connect(multimeter, sensory_nodes)

    # Create Poisson generator for background activity
    spike_gen = None
    if bg_rate is not None:
        rate = (bg_rate / ff_factor) * p_rf
        spike_gen = nest.Create("poisson_generator", n=num_neurons, params={"rate": rate})

        syn_spec = {"weight": synaptic_strength}
        conn_spec = {"rule": "one_to_one"}
        nest.Connect(spike_gen, sensory_nodes, conn_spec=conn_spec, syn_spec=syn_spec)

    return torus_layer, spikedetector, multimeter, spike_gen


def create_torus_layer_with_jitter(
        num_neurons=3600,
        jitter=0.03,
        neuron_type="iaf_psc_alpha",
        layer_size=R_MAX
):
    """
    Create a layer wrapped a torus to avoid boundary conditions. Neurons are placed on a grid with fluctuations
    :param num_neurons: Number of neurons in layer
    :param jitter: amount of jitter
    :param neuron_type: Type of the neurons. All neuron models are supported
    :param layer_size: Size of the layer
    :return: layer
    """
    # Create coordinates of neurons
    mod_size = layer_size - jitter*2
    step_size = mod_size / float(np.sqrt(num_neurons))
    coordinate_scale = np.arange(-mod_size / 2., mod_size / 2., step_size)
    grid = [[x, y] for y in coordinate_scale for x in coordinate_scale]
    positions = [[pos[0] + np.random.uniform(-jitter, jitter),
                 pos[1] + np.random.uniform(-jitter, jitter)]
                 for pos in grid]

    # Create dict for neural layer that is wrapped as torus to avoid boundary effects
    torus_dict = {
        "extent": [layer_size, layer_size],
        "positions": positions,
        "elements": neuron_type,
        "edge_wrap": True
    }

    # Create layer
    torus_layer = tp.CreateLayer(torus_dict)
    return torus_layer


def create_random_connections(
        layer,
        inh_neurons,
        connect_dict=None,
        cap_s=1.,
        inh_weight=-1.,
        plot=False,
        save_plot=False,
        save_prefix="",
        color_mask=None
):
    """
    Establish random connections
    :param layer: The neural layer
    :param inh_neurons: The ids of the inhibitory neurons
    :param connect_dict: The connection dict that specifies the connection rules
    :param cap_s: The excitatory weight
    :param inh_weight: The inhibitory weight
    :param plot: Flag to determine whether a plot should be created
    :param save_plot: Flag to determine whether the plot should be saved. If set to False it is displayed instead
    :param save_prefix: An additional naming prefix that can be set if the plot is saved
    :param color_mask: Optional tuning map that can be used to color the background of the neurons
    :return: None
    """
    # Define connection parameters
    nodes = nest.GetNodes(layer, properties={"element_type": "neuron"})[0]
    exc_nodes = list(set(nodes).difference(inh_neurons))
    if connect_dict is None:
        connect_dict = {
            "rule": "pairwise_bernoulli",
            "p": 0.01
        }
    nest.Connect(exc_nodes, nodes, conn_spec=connect_dict, syn_spec={"weight": cap_s})
    if type(inh_neurons) == np.ndarray:
        inh_neurons = inh_neurons.tolist()
    nest.Connect(inh_neurons, nodes, conn_spec=connect_dict, syn_spec={"weight": -abs(inh_weight)})

    if plot:
        layer_size = nest.GetStatus(layer, "topology")[0]["extent"][0]
        nodes = nest.GetNodes(layer)[0]
        node = nodes[0]
        node_conn = nest.GetConnections([node], target=nodes)
        target_nodes = nest.GetStatus(node_conn, "target")
        plot_connections(
            [node],
            target_nodes,
            layer_size=layer_size,
            color_mask=color_mask,
            save_plot=save_plot,
            save_prefix=save_prefix,
            plot_name="random_local_connections.png"
        )


def create_local_circular_connections_topology(
        layer,
        r_loc=0.5,
        p_loc=0.7,
        allow_autapses=False,
        allow_multapses=False,
        plot=False,
        save_plot=False,
        color_mask=None
):
    """
    Create local connections with in a circular radius
    :param layer: The layer where the local connections should be established
    :param r_loc: radius for local connections
    :param p_loc: probability of establishing local connections
    :param allow_autapses: Flag to allow self-connections
    :param allow_multapses: Flag to allow multiple connections between neurons
    :param plot: Flag for plotting connections
    :param save_plot: Flag for saving the plot. If not plotted the parameter is ignored
    :param color_mask: Color/orientation map for neurons. If not plotted the parameter is ignored
    """

    # Define mask
    mask_dict = {
        "circular": {"radius": r_loc}
    }

    # Define connection parameters
    connection_dict = {
        "connection_type": "divergent",
        "mask": mask_dict,
        "kernel": p_loc,
        "allow_autapses": allow_autapses,
        "allow_multapses": allow_multapses,
        "synapse_model": "static_synapse"
    }

    tp.ConnectLayers(layer, layer, connection_dict)

    if plot:
        node = nest.GetNodes(layer)[0]
        node_conn = nest.GetConnections([node])
        target_nodes = nest.GetStatus(node_conn, "targets")
        plot_connections(
            [node],
            target_nodes,
            color_mask=color_mask,
            save_plot=save_plot,
            plot_name="circular_local_connections.png"
        )


def create_local_circular_connections(
        layer,
        node_tree,
        inh_neurons,
        inh_weight=-1.,
        r_loc=0.5,
        cap_s=1.,
        connect_dict=None,
        plot=False,
        save_plot=False,
        save_prefix="",
        color_mask=None
):
    """
    Create local connections with in a circular radius
    :param layer: The layer where the local connections should be established
    :param node_tree: Positons of the nodes organised in a tree
    :param inh_neurons: IDs of the inhibitory neurons
    :param inh_weight: The weight for inhibitory connections
    :param r_loc: radius for local connections
    :param cap_s: The weight for excitatory connections
    :param connect_dict: Dictionary that specifies the connection rules
    :param plot: Flag for plotting connections
    :param save_plot: Flag for saving the plot. If not plotted the parameter is ignored
    :param save_prefix: Naming prefix that can be used if the plot is saved
    :param color_mask: Color/orientation map for neurons. If not plotted the parameter is ignored
    :return None
    """

    if connect_dict is None:
        connect_dict = {
            "rule": "pairwise_bernoulli",
            "p": 0.7,
        }

    nodes = nest.GetNodes(layer, properties={"element_type": "neuron"})[0]
    positions = tp.GetPosition(nodes)
    for n, pos in zip(nodes, positions):
        connect_partners = (np.asarray(node_tree.query_ball_point(pos, r_loc)) + min(nodes)).tolist()
        if n not in inh_neurons:
            syn_dict = {"weight": cap_s}
        else:
            syn_dict = {"weight": inh_weight}
        nest.Connect([n], connect_partners, conn_spec=connect_dict, syn_spec=syn_dict)

        # Plot first one
        if n - min(nodes) == 0:
            if plot:
                # Assume that layer is a square
                layer_size = nest.GetStatus(layer, "topology")[0]["extent"][0]
                connect = nest.GetConnections([n])
                targets = nest.GetStatus(connect, "target")
                local_targets = [t for t in targets if t in list(connect_partners)]
                plot_connections(
                    [n],
                    local_targets,
                    layer_size=layer_size,
                    save_plot=save_plot,
                    plot_name="circular_local_connections.png",
                    save_prefix=save_prefix,
                    color_mask=color_mask,
                )


def create_distant_np_connections(
        layer,
        r_loc=0.5,
        p_loc=0.7,
        p_p = None,
        allow_multapses=False
):
    """
    Create long distance connections without any patches
    :param layer: Layer in which the connections should be established
    :param r_loc: Radius for local connections needed to calculate the long range connection probability
    :param p_loc: Probability for local connections needed to calculate the long range connection probability
    :param p_p: Probability to establish long-range patchy connections. If none, prob. is calculated according to Voges
                paper
    :param allow_multapses: Allow multiple connections between neurons
    :return Neurons of the layer for debugging (plotting)
    """
    layer_size = nest.GetStatus(list(layer), "topology")[0]["extent"][0]
    # Mask for area to which long-range connections can be established
    mask_dict = {
        "doughnut": {
            "inner_radius": r_loc,
            "outer_radius": layer_size / 2.
        }
    }

    if p_p is None:
        # Get possibility to establish a single long-range connection
        p_p = get_lr_connection_probability_np(r_loc, p_loc, layer_size=layer_size)

    connection_dict = {
        "connection_type": "divergent",
        "mask": mask_dict,
        "kernel": p_p,
        "allow_autapses": False,
        "allow_multapses": allow_multapses,
        "allow_oversized_mask": True,
    }

    tp.ConnectLayers(layer, layer, connection_dict)

    # Return nodes of layer for debugging
    return nest.GetNodes(layer)[0]


def create_random_patches(
        layer,
        inh_neurons,
        r_loc=0.5,
        p_loc=0.7,
        num_patches=3,
        r_p=0.25,
        cap_s=1.,
        p_p=None,
        plot=True,
        save_plot=False,
        save_prefix="",
        color_mask=None
):
    """
    Create random long range patchy connections. To every neuron a single link is established instead of
    taking axonal morphology into account.
    :param layer: Layer in which the connections should be established
    :param inh_neurons: The IDs of the inhibitory neurons
    :param r_loc: Radius for local connections
    :param p_loc: Probability for local connections
    :param num_patches: Number of patches that should be created
    :param cap_s: The excitatory weight
    :param p_p: Probability to establish long-range patchy connections. If none, prob. is calculated according to Voges
    paper
    :param plot: Flag to determine whether the connections of an example should be plotted
    :param save_plot: Flag to determine whether the plot should be saved. Is ignored if the plot flag is set to False
    :param save_prefix: Naming prefix for saved plots
    :param color_mask: The color mask for the tuning areas of neurons
    :return Nodes of the layer for debugging purposes (plotting)
    """
    # Calculate the parameters for the patches
    layer_size = nest.GetStatus(layer, "topology")[0]["extent"][0]
    if r_p is None:
        r_p = r_loc / 2.
    min_distance = r_loc + r_p
    max_distance = layer_size / 2. - r_loc
    if p_p is None:
        p_p = get_lr_connection_probability_patches(r_loc, p_loc, r_p, num_patches=num_patches, layer_size=layer_size)

    # Iterate through all neurons, as all neurons have random patches
    nodes = nest.GetNodes(layer, properties={"element_type": "neuron"})[0]
    # Do not establish lr connections with inh neurons
    exc_nodes = list(set(nodes).difference((set(inh_neurons))))
    for neuron in exc_nodes:
        # Calculate radial distance and the respective coordinates for patches
        radial_angle = np.random.uniform(0., 359., size=num_patches).tolist()
        radial_distance = np.random.uniform(min_distance, max_distance, size=num_patches).tolist()

        # Calculate patch region
        mask_specs = {"radius": r_p}
        anchors = [to_coordinates(angle, distance) for angle, distance in zip(radial_angle, radial_distance)]
        patches = tuple()
        for anchor in anchors:
            anchor = (np.asarray(tp.GetPosition([neuron])[0]) + np.asarray(anchor)).tolist()
            patches += tp.SelectNodesByMask(layer, anchor, mask_obj=tp.CreateMask("circular", specs=mask_specs))

        # Define connection
        connect_dict = {
            "rule": "pairwise_bernoulli",
            "p": p_p
        }

        syn_spec = {"weight": cap_s}
        nest.Connect([neuron], patches, conn_spec=connect_dict, syn_spec=syn_spec)

    if plot:
        nodes = nest.GetNodes(layer)[0]
        node = nodes[0]
        node_conn = nest.GetConnections(source=[node], target=nodes)
        target_nodes = nest.GetStatus(node_conn, "target")
        plot_connections(
            [node],
            target_nodes,
            layer_size=layer_size,
            color_mask=color_mask,
            save_plot=save_plot,
            save_prefix=save_prefix,
            plot_name="random_lr_patches.png"
        )
    # Return nodes of layer for debugging
    return nest.GetNodes(layer)[0]


def create_overlapping_patches(
        layer,
        r_loc=0.5,
        p_loc=0.7,
        distance=2.5,
        num_patches=3,
        allow_multapses=False,
        p_p=None
):
    """
    Create long-range patchy connections with overlapping patches such that all neurons share almost the same
    parameters for these links
    :param layer: The layer in which the connections should be established
    :param r_loc: Radius of local connections
    :param p_loc: Probability of local connections
    :param distance: Distance of patches
    :param num_patches: Number of patches
    :param allow_multapses: Flag to allow multiple links between neurons
    :param p_p: Probability to establish long-range patchy connections. If none, prob. is calculated according to Voges
                paper
    :return: Neurons of the layer for debugging (plotting)
    """
    layer_size = nest.GetStatus(layer, "topology")[0]["extent"][0]
    # Calculate parameters for pathces
    r_p = r_loc / 2.
    if p_p is None:
        p_p = get_lr_connection_probability_patches(r_loc, p_loc, r_p, layer_size=layer_size)

    # Create overlapping patches as every neuron shares the same patch parameter
    for n in range(1, num_patches+1):
        angle = n * 360. / float(num_patches)
        coordinates = to_coordinates(angle, distance)
        mask_dict = {
            "circular": {"radius": r_p, "anchor": coordinates}
        }

        connection_dict = {
            "connection_type": "divergent",
            "mask": mask_dict,
            "kernel": p_p,
            "allow_autapses": False,
            "allow_multapses": allow_multapses,
            "synapse_model": "static_synapse"
        }

        tp.ConnectLayers(layer, layer, connection_dict)

    # Return nodes of layer for debugging
    return nest.GetNodes(layer)[0]


def create_shared_patches(
        layer,
        r_loc=0.5,
        p_loc=0.7,
        size_boxes=0.5,
        num_patches=3,
        num_shared_patches=6,
        p_p=None
):
    """
    Create shared patches for long-range connections
    :param layer: Layer in which the connections should be established
    :param r_loc: Radius of local connections
    :param p_loc: Probility of local connections
    :param size_boxes: Size of a sublayer that defines the set of possible patches a neuron in this patch can create
    links to
    :param num_patches: Number of patches per neuron
    :param num_shared_patches: Number of patches per sublayer
    :param p_p: Probability to establish long-range patchy connections. If none, prob. is calculated according to Voges
                paper
    :return: Neurons of sublayer anchored at size_boxes/2 for debugging (plotting)
    """
    # Number of patches per neuron must be lower or equal to the number of patches per box, as the patches of a neuron
    # are a subset of the patches of the sublayer
    assert num_patches <= num_shared_patches

    return _create_location_based_patches(
        layer=layer,
        r_loc=r_loc,
        p_loc=p_loc,
        size_boxes=size_boxes,
        num_patches=num_patches,
        num_shared_patches=num_shared_patches,
        is_partially_overlapping=False,
        p_p=p_p
    )


def create_partially_overlapping_patches(
        layer,
        r_loc=0.5,
        p_loc=0.7,
        size_boxes=0.5,
        num_patches=3,
        num_shared_patches=6,
        num_patches_replaced=3,
        p_p=None
):
    """
    Create partially overlapping patches for long-range connections
    :param layer: Layer in which the connections should be established
    :param r_loc: Radius of local connections
    :param p_loc: Probability of local connections
    :param size_boxes: Size of a sublayer that defines the set of possible patches a neuron in this patch can create
    links to
    :param num_patches: Number of patches per neuron
    :param num_shared_patches: Number of patches per sublayer
    :param num_patches_replaced: Number of patches that are replaced for every box in x-direction
    :param p_p: Probability to establish long-range patchy connections. If none, prob. is calculated according to Voges
                paper
    :return: Neurons of the sublayer anchored at box_size/2 for debugging (plotting)
    """
    assert num_patches_replaced <= num_shared_patches
    assert num_patches <= num_shared_patches

    return _create_location_based_patches(
        layer=layer,
        r_loc=r_loc,
        p_loc=p_loc,
        size_boxes=size_boxes,
        num_patches=num_patches,
        num_shared_patches=num_shared_patches,
        num_patches_replaced=num_patches_replaced,
        is_partially_overlapping=True,
        p_p=p_p
    )


def create_stimulus_based_local_connections(
        layer,
        node_tree,
        tuning_vector,
        tuning_to_neuron_map,
        inh_nodes,
        inh_weight=-1.,
        r_loc=0.5,
        cap_s=1.,
        connect_dict=None,
        plot=False,
        save_plot=False,
        save_prefix="",
        color_mask=None
):
    """
    Create local connections only to neurons with same stimulus feature preference
    :param layer: Layer with neurons
    :param node_tree: Node positions organised in a tree
    :param tuning_vector: Vector with tuning classes
    :param tuning_to_neuron_map: Dictionary mapping from tuning preference to all neurons with that preference
    :param inh_nodes: IDs of inhibitory nodes
    :param inh_weight: Weight of the inhibitory connections
    :param r_loc: Radius within which local connections can be established
    :param cap_s: Excitatory weight
    :param connect_dict: Connection dictionary specifying the connection parameter
    :param plot: Flag for plotting
    :param save_plot: Flag for saving the plot. If plot is False this parameter is ignored
    :param save_prefix: Naming prefix for the saved plots. Is ignored if the plot is not saved
    :param color_mask: Color/orientation map for plotting. If plot is False this parameter is ignored
    :return None
    """
    if connect_dict is None:
        connect_dict = {
            "rule": "pairwise_bernoulli",
            "p": 0.7,
        }

    node_ids = nest.GetNodes(layer, properties={"element_type": "neuron"})[0]
    min_node_id = min(node_ids)
    node_pos = tp.GetPosition(node_ids)
    for node, pos in zip(node_ids, node_pos):
        connect_partners = (np.asarray(node_tree.query_ball_point(pos, r_loc)) + min_node_id).tolist()
        if node not in inh_nodes:
            stimulus_tuning = tuning_vector[node - min_node_id]
            similar_tuned_neurons = tuning_to_neuron_map[stimulus_tuning]
            same_stimulus_area = set(filter(lambda n: n != node, similar_tuned_neurons))
            connect_partners = list(set(connect_partners).intersection(same_stimulus_area))
            syn_spec = {"weight": float(cap_s)}
        else:
            syn_spec = {"weight": float(-abs(inh_weight))}

        nest.Connect([node], connect_partners, conn_spec=connect_dict, syn_spec=syn_spec)

        # Plot first one
        if node - min_node_id == 0:
            if plot:
                # Assume that layer is a square
                layer_size = nest.GetStatus(layer, "topology")[0]["extent"][0]
                connect = nest.GetConnections([node])
                targets = nest.GetStatus(connect, "target")
                local_targets = [t for t in targets if t in list(connect_partners)]
                plot_connections(
                    [node],
                    local_targets,
                    layer_size=layer_size,
                    save_plot=save_plot,
                    save_prefix=save_prefix,
                    plot_name="stimulus_dependent_local_connections.png",
                    color_mask=color_mask,
                )


def create_stimulus_based_patches_random(
        layer,
        tuning_vector,
        tuning_to_neuron_map,
        inh_neurons,
        node_tree,
        num_patches=2,
        r_loc=0.5,
        p_loc=0.7,
        p_p=0.7,
        cap_s=1.,
        r_p=None,
        connect_dict=None,
        filter_patches=True,
        plot=False,
        save_plot=False,
        save_prefix="",
        color_mask=None
):
    """
    Create patchy connections based on tuning preference
    :param layer: Layer with neurons
    :param tuning_vector: Vector with tuning classes
    :param tuning_to_neuron_map: Dictionary mapping from tuning preference to all neurons with that preference
    :param inh_neurons: IDs of inhibitory neurons
    :param node_tree: Node positions organised in a tree
    :param num_patches: Number of patches that are created
    :param r_loc: Radius for local connections
    :param p_loc: Probability for local connections. If p_p or the connect dict is not None this parameter is ignored
    :param p_p: Probability for patchy connections
    :param cap_s: Excitatory weight
    :param r_p: Patchy radius. If None it is half of the local radius
    :param connect_dict: Dictionary specifying the connections that are to establish
    :param filter_patches: If set to True only long-range connections are established to neurons with the same stimulus
    preference in the radius around the center
    :param plot: Flag for plotting
    :param save_plot: Flag for saving the plot. If plot is set to False this parameter is ignored
    :param save_prefix: Naming prefix for the saved plot. Ignored if plot is not saved
    :param color_mask: Color/orientation map. If plot is set to False this parameter is ignored
    :return None
    """
    size_layer = nest.GetStatus(layer, "topology")[0]["extent"][0]
    # Calculate parameters for the patches
    if connect_dict is None and p_p is None:
        warnings.warn("Connect dict not specified.\n"
                      "Computed probability based on Voges does not necessarily return in all cases a valid value.\n"
                      "Recommended values are p_loc >= 0.5 and r_loc > 0.5")

    if r_p is None:
        r_p = r_loc / 2.

    min_distance = r_loc + r_p
    max_distance = np.sqrt(size_layer**2 + size_layer**2) / 2. - r_p
    if p_p is None and connect_dict is None:
        p_p = get_lr_connection_probability_patches(r_loc, p_loc, r_p, num_patches=num_patches, layer_size=size_layer)
    node_ids = nest.GetNodes(layer, properties={"element_type": "neuron"})[0]

    # Do not establish lr connections for inh neurons
    min_id = min(node_ids)
    exc_neurons = list(set(node_ids).difference(set(inh_neurons)))
    exc_pos = tp.GetPosition(exc_neurons)
    for neuron, pos in zip(exc_neurons, exc_pos):
        inner_nodes = (np.asarray(node_tree.query_ball_point(pos, min_distance)) + min(node_ids)).tolist()
        outer_nodes = (np.asarray(node_tree.query_ball_point(pos, max_distance)) + min(node_ids)).tolist()
        patchy_candidates = set(outer_nodes).difference(set(inner_nodes))
        stimulus_tuning = tuning_vector[neuron - min_id]
        same_tuning_nodes = tuning_to_neuron_map[stimulus_tuning]
        patchy_candidates = patchy_candidates.intersection(set(same_tuning_nodes))

        # TODO is sorting by angle needed at some point?
        # pos_n = tp.GetPosition([neuron])[0]
        # pos_patchy_candidates = tp.GetPosition(patchy_candidates)
        # angles = np.arctan2(
        #     np.asarray(pos_patchy_candidates)[:, 1] - pos_n[1],
        #     np.asarray(pos_patchy_candidates)[:, 0] - pos_n[0]
        # ).reshape(-1) / np.pi * 180.
        # patchy_candidates_angles = sorted(zip(patchy_candidates, angles), key=lambda stn: stn[1], reverse=True)
        # patchy_candidates, _ = zip(*patchy_candidates_angles)

        patch_center_nodes = np.random.choice(
            list(patchy_candidates),
            size=np.minimum(len(patchy_candidates), num_patches),
            replace=False
        ).tolist()
        pos_patch_centers = tp.GetPosition(patch_center_nodes)

        if len(patch_center_nodes) < num_patches:
            warnings.warn(
                "Did not find enough patches for neuron %s: found only %s instead of %s"
                % (neuron, len(patch_center_nodes), num_patches)
            )

        lr_patches = tuple()
        for neuron_anchor in pos_patch_centers:
            lr_patches += tuple((np.asarray(node_tree.query_ball_point(neuron_anchor, r_p)) + min(node_ids)).tolist())

        if filter_patches:
            lr_patches = tuple(set(lr_patches).intersection(set(same_tuning_nodes)))

        # Define connection
        if connect_dict is None:
            connect_dict = {
                "rule": "pairwise_bernoulli",
                "p": p_p
            }
        syn_spec = {"weight": cap_s}

        nest.Connect([neuron], lr_patches, conn_spec=connect_dict, syn_spec=syn_spec)

        # Plot only first neuron
        if neuron - min(exc_neurons) == 0:
            connect = nest.GetConnections([neuron])
            targets = nest.GetStatus(connect, "target")
            patchy_targets = [t for t in targets if t in list(lr_patches)]
            if plot:
                plot_connections(
                    [neuron],
                    patchy_targets,
                    size_layer,
                    color_mask=color_mask,
                    save_plot=save_plot,
                    save_prefix=save_prefix,
                    plot_name="stimulus_dependent_lr_patchy_connections.png"
                )


def set_synaptic_strength(
        nodes,
        adj_mat,
        cap_s=1.,
        divide_by_num_connect=False
):
    """
    Set the synaptic strength. It can be made dependent on number of established connections if needed
    :param nodes: Nodes for which the connections should be adapted
    :param adj_mat: Adjacency matrix for the connections. NOTE: the adjacency matrix isn't updated as it is assumed to
    not be a weight matrix
    :param cap_s: Weight for the connections
    :param divide_by_num_connect: Flag for dividing the weight through the number of established connections
    :return None
    """
    num_connec = adj_mat.sum()
    if num_connec == 0:
        return
    connect = nest.GetConnections(source=nodes)
    connect = [c for c in connect if nest.GetStatus([c], "target")[0] in nodes]
    weight = cap_s / float(num_connec) if divide_by_num_connect else cap_s
    nest.SetStatus(connect, {"weight": weight})


def create_input_current_generator(
        input_stimulus,
        organise_on_grid=False,
        multiplier=1.
):
    """
    Create direct current generator to simulate input stimulus. The pixel values of the image are transformed
    to an integer value representing the intensity in Ampere A
    :param input_stimulus: Grayscale input stimulus with integer values between 0 and 256
    :param organise_on_grid: Flag for organising current generators on a grid to represent receptors
    :param multiplier: factor that is multiplied to the injected current
    :return: Tuple with ids for the dc generator devices
    """
    assert np.all(input_stimulus < 256) and np.all(input_stimulus >= 0)

    num_receptors = input_stimulus.size
    # Multiply value with 1e12, as the generator expects values in pA
    current_dict = [{"amplitude": float(amplitude) * multiplier} for amplitude in input_stimulus.reshape(-1)]
    if not organise_on_grid:
        dc_generator = nest.Create("dc_generator", n=int(num_receptors), params=current_dict)
    else:
        # Note that PyNest uses [column, row]
        receptor_layer_dict = {
            "extent": [float(input_stimulus.shape[1]), float(input_stimulus.shape[0])],
            "rows": input_stimulus.shape[0],
            "columns": input_stimulus.shape[1],
            "elements": "dc_generator",
        }
        dc_generator = tp.CreateLayer(receptor_layer_dict)
        dc_nodes = nest.GetNodes(dc_generator)[0]
        nest.SetStatus(dc_nodes, current_dict)

    return dc_generator


def create_sensory_nodes(
        num_neurons=1e3,
        time_const=20.0,
        rest_pot=0.0,
        threshold_pot=1e3,
        capacitance=1e12,
        use_barranca=True
):
    """
    Create the sensory nodes of the network
    :param num_neurons: Number of sensory nodes that have to be created
    :param time_const: Membrane time constant in ms
    :param rest_pot: Resting potential / reset potential in mV
    :param threshold_pot: Threshold potential in mV
    :param capacitance: Capacitance of the membrane in pF
    :param use_barranca: Flag determining whether the customised Barranca neuron should be used
    :return: Tuple with ids of the neurons
    """
    if use_barranca:
        neuron_dict_barranca = {
            "tau_m": time_const,
            "V_th": threshold_pot,
            "V_R": rest_pot,
            "C_m": capacitance
        }
        sensory_nodes = nest.Create("barranca_neuron", n=int(num_neurons), params=neuron_dict_barranca)
    else:
        neuron_dict_iaf_delta = {
            "V_m": rest_pot,
            "E_L": rest_pot,
            "C_m": capacitance,
            "tau_m": time_const,
            "V_th": threshold_pot,
            "V_reset": rest_pot
        }
        sensory_nodes = nest.Create("iaf_psc_delta", n=int(num_neurons), params=neuron_dict_iaf_delta)

    # Create spike detector for stimulus reconstruction
    spikedetector = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
    multimeter = nest.Create("multimeter", params={"withtime": True, "record_from": ["V_m"]})
    nest.Connect(sensory_nodes, spikedetector)
    nest.Connect(multimeter, sensory_nodes)
    return sensory_nodes, spikedetector, multimeter


def create_connections_random(
        src_nodes,
        target_nodes,
        indegree=10,
        connection_strength=0.7,
):
    """
    Create synaptic connections from source to target nodes that are not limited in the area (i.e. the function does
    not consider receptive fields)
    :param src_nodes: Source nodes
    :param target_nodes: Target nodes
    :param indegree: Number of pre-synaptic nodes per neuron
    :param connection_strength: Synaptic weight of the connections
    :return None
    """
    connect_dict = {
        "rule": "fixed_indegree",
        "indegree": indegree
    }

    synapse_dict = {
        "weight": connection_strength
    }

    nest.Connect(src_nodes, target_nodes, conn_spec=connect_dict, syn_spec=synapse_dict)


def create_perlin_stimulus_map(
        layer,
        inh_neurons,
        num_stimulus_discr=4,
        resolution=(10, 10),
        spacing=0.1,
        plot=False,
        save_plot=False,
        save_prefix=""
):
    """
    Create stimulus map that is based on the Perlin noise distribution
    :param layer: Neural layer
    :param inh_neurons: IDs of inhibitory neurons
    :param num_stimulus_discr: Number of stimulus feature classes that can be discriminated
    :param resolution: Perlin noise resolution defines the mesh of randomly created values and vectors
    :param spacing: Size of a single cell in x and y direction of the mesh for the interpolated values
    :param plot: If set to True the tuning map is plotted
    :param save_plot: If set to True the tuning map is saved. This is ignored if plot is set to False
    :param save_prefix: Naming prefix of the saved plot
    :return: Tuning to neuron map, neuron to tuning map, color map
    """
    size_layer = nest.GetStatus(layer, "topology")[0]["extent"][0]
    nodes = nest.GetNodes(layer, properties={"element_type": "neuron"})[0]

    tuning_to_neuron_map = {stimulus: [] for stimulus in range(num_stimulus_discr)}
    tuning_vector = np.zeros(len(nodes))

    c_map = perlin_noise(size_layer, resolution=resolution, spacing=spacing)

    ind = np.indices(c_map.shape)
    # Zip the row and column indices
    ind = list(zip(ind[0].reshape(-1), ind[1].reshape(-1)))
    c_map_sorted = sorted(zip(c_map.reshape(-1), ind), key=lambda x: x[0])
    _, ind = zip(*c_map_sorted)
    step_size = len(c_map_sorted) // num_stimulus_discr
    color_map = -1 * np.ones(c_map.shape, dtype='int')
    for stim_class in range(num_stimulus_discr):
        row, col = zip(*ind[stim_class*step_size: np.minimum(len(ind)-1, (stim_class+1)*step_size)])
        color_map[row, col] = int(stim_class)

    color_map[c_map == c_map.max()] = num_stimulus_discr - 1

    positions = np.asarray(tp.GetPosition(nodes))
    inh_mask = np.zeros(len(nodes)).astype('bool')
    inh_mask[np.asarray(inh_neurons) - min(nodes)] = True
    x_grid, y_grid = coordinates_to_cmap_index(size_layer, positions[~inh_mask], spacing)
    stim_class = color_map[x_grid, y_grid]
    zip_node_class = list(zip(np.asarray(nodes)[~inh_mask].tolist(), stim_class.tolist()))
    tuning_vector[~inh_mask] = stim_class
    tuning_vector[inh_mask] = -1
    for c in range(num_stimulus_discr):
        stimulus_class_list = list(filter(lambda x: x[1] == c, zip_node_class))
        stim_nodes, _ = zip(*stimulus_class_list)
        tuning_to_neuron_map[c] = stim_nodes

    if plot:
        plot_cmap(
            ff_nodes=nodes,
            inh_nodes=inh_neurons,
            color_map=color_map,
            stim_class=stim_class,
            positions=positions,
            size_layer=size_layer,
            resolution=resolution,
            num_stimulus_discr=num_stimulus_discr,
            save_plot=save_plot,
            save_prefix=save_prefix
        )

    return tuning_to_neuron_map, tuning_vector, color_map


def create_stimulus_tuning_map(
        layer,
        num_stimulus_discr=4,
        stimulus_per_row=2,
        plot=False,
        save_plot=False,
        plot_name=None,
        save_prefix=""
):
    """
    Create the stimulus tuning map for neurons
    :param layer: Layer of neurons
    :param num_stimulus_discr: Number of stimuli feature classes that are to discriminate
    :param stimulus_per_row: Number of how often a stimulus should be represented per row
    :param plot: Flag for plotting
    :param save_plot: Flag for saving the plot. If plot is set to False this parameter is ignored
    :param plot_name: Name of the saved plot file. If plot is set to False this parameter is ignored
    :param save_prefix: Naming prefix that can be set before the plot name. If plot or save_plot is set to False this
    parameter is ignored
    :return: Map from stimulus feature class to neuron, map form neuron to stimulus feature class, vector with weights
            for their respective class, e.g. class of neurons / # classes
    """
    size_layer = nest.GetStatus(layer, "topology")[0]["extent"][0]
    # Assume quadratic layer size
    box_size = size_layer / float(stimulus_per_row * num_stimulus_discr)
    sublayer_anchors, box_mask_dict = create_distinct_sublayer_boxes(size_boxes=box_size, size_layer=size_layer)
    tuning_to_neuron_map = {stimulus: [] for stimulus in range(num_stimulus_discr)}
    neuron_to_tuning_map = {}
    nodes = nest.GetNodes(layer, properties={"element_type": "neuron"})[0]
    tuning_weight_vector = np.zeros(len(nodes))
    min_idx = min(nodes)
    shift = 0
    color_map = []
    for num, anchor in enumerate(sublayer_anchors):
        stimulus_area = tp.SelectNodesByMask(layer, anchor, mask_obj=tp.CreateMask("rectangular", specs=box_mask_dict))
        # TODO correct error handling?
        if len(stimulus_area) == 0:
            continue
        stimulus_tuning = (num + shift) % num_stimulus_discr
        if num % num_stimulus_discr == num_stimulus_discr - 1:
            shift = 0 if shift == num_stimulus_discr // 2 else num_stimulus_discr // 2
        # Note that every value is a list of tuples
        tuning_to_neuron_map[stimulus_tuning].extend(stimulus_area)
        for neuron in stimulus_area:
            tuning_weight_vector[neuron - min_idx] = stimulus_tuning / float(num_stimulus_discr - 1)
        neuron_to_tuning_sub = {neuron: stimulus_tuning for neuron in stimulus_area}
        neuron_to_tuning_map = {**neuron_to_tuning_map, **neuron_to_tuning_sub}

        color = list(mcolors.TABLEAU_COLORS)[stimulus_tuning]
        color_map.append({
            "lower_left": (anchor[0] + box_mask_dict["lower_left"][0], anchor[1] + box_mask_dict["lower_left"][1]),
            "width": box_mask_dict["upper_right"][0] - box_mask_dict["lower_left"][0],
            "height": box_mask_dict["upper_right"][1] - box_mask_dict["lower_left"][1],
            "color": color
        })

        if plot:

            positions = tp.GetPosition(stimulus_area)
            x, y = zip(*positions)
            plt.plot(x, y, '.', color=color)
            area_rect = patches.Rectangle(
                color_map[-1]["lower_left"],
                width=color_map[-1]["width"],
                height=color_map[-1]["height"],
                color=color_map[-1]["color"],
                alpha=0.4
            )
            plt.gca().add_patch(area_rect)
            plt.plot()

    if plot:
        if not save_plot:
            plt.show()
            plt.close()
        else:
            curr_dir = os.getcwd()
            Path(curr_dir + "/figures/tuning-map/").mkdir(parents=True, exist_ok=True)
            if plot_name is None:
                plot_name = "stimulus_tuning_map_squared.png"
            plt.savefig(curr_dir + "/figures/tuning-map/%s_%s" % (save_prefix, plot_name))
            plt.close()

    return tuning_to_neuron_map, neuron_to_tuning_map, tuning_weight_vector, color_map


def step_tuning_curve(
        input_stimulus,
        stimulus_tuning,
        tuning_discr_steps,
        multiplier=1.
):
    """
    Function to check whether a certain neuron reacts on a certain stimulus (e.g. stimulus is within the stimulus
    feature class)
    :param input_stimulus: The input stimulus
    :param stimulus_tuning: The tuning of the neuron (identifier for the class)
    :param tuning_discr_steps: The number of discriminated stimulus features
    :param multiplier: Factor that is multiplied to the injected current
    :return: True if neuron reacts, False otherwise
    """
    addent = 0 if (stimulus_tuning + 1) * tuning_discr_steps != 255. else 1
    return np.logical_and(
        stimulus_tuning * tuning_discr_steps <= input_stimulus / multiplier,
        input_stimulus / multiplier < (stimulus_tuning + 1) * tuning_discr_steps + addent
    )


def continuous_tuning_curve(
        input_stimulus,
        stimulus_tuning,
        tuning_discr_steps,
        sigma=None,
        max_value=255.
):
    """
    Gaussian shaped tuning function
    :param input_stimulus: Input from the recpetive field
    :param stimulus_tuning: Tuning class
    :param tuning_discr_steps: Total number of tuning classes
    :param sigma: Standard derivation
    :param max_value: Maximal value of the peak
    :return: Tuning
    """
    sigma = sigma if sigma is not None else tuning_discr_steps
    mu = (stimulus_tuning + 1) * tuning_discr_steps

    return max_value * np.exp((input_stimulus - mu)**2 / float(-2 * sigma**2))


def linear_tuning(
        input_stimulus,
        stimulus_tuning,
        tuning_discr_steps
):
    """
    Linear tuning function
    :param input_stimulus: Input stimulus from the recpetive fields
    :param stimulus_tuning: Tuning class
    :param tuning_discr_steps: Total number of tuning classes
    :return: Tuning
    """
    sign = -1. if stimulus_tuning % 2 == 0 else 1.
    intercept = 2 * sign * stimulus_tuning * tuning_discr_steps
    slope = -sign * int(stimulus_tuning / 2 + 1)
    return slope * input_stimulus + intercept, slope, intercept


def _set_input_current(neuron, current_dict, synaptic_strength, use_dc=True):
    """
    Set the input current or the spiking rate
    :param neuron: Neuron for which the input is set
    :param current_dict: Defining current specification
    :param synaptic_strength: Weight
    :param use_dc: If set to True a DC generator is used, otherwise a Poisson generator
    :return:
    """
    connections = nest.GetConnections()
    sources = [c[0] for c in connections if c[1] == neuron]
    source_types = np.asarray(list(nest.GetStatus(sources, "element_type")))
    generator = np.array([])
    if len(source_types) > 0:
        generator = np.asarray(sources)[source_types == "stimulator"]

    if generator.size == 0 or use_dc:
        if use_dc:
            generator = nest.Create("dc_generator", n=1, params=current_dict)[0]
        else:
            generator = nest.Create("poisson_generator", n=1, params=current_dict)[0]
        syn_spec = {"weight": synaptic_strength}
        nest.Connect([generator], [neuron], syn_spec=syn_spec)
    else:
        nest.SetStatus(generator.tolist(), current_dict)

    return generator


def same_input_current(layer, synaptic_strength, connect_prob, value=255/2., rf_size=(10, 10), use_dc=False):
    """
    Set the same injected input (either Poisson spike train or direct current) for all sensory neurons
    :param layer: Layer with sensory neurons
    :param synaptic_strength: Synaptic weights
    :param connect_prob: Connection probability
    :param value: The value to which it is set
    :param rf_size: Size of the receptive field
    :param use_dc: If set to true, direct current is injected
    :return: List of generaturs that inject input
    """
    if use_dc:
        current_dict = {"amplitude": (rf_size[0] * rf_size[1]) * value * connect_prob}
    else:
        rate = 1000. * value * connect_prob / 255.
        current_dict = {"rate": rate}
    neurons = nest.GetNodes(layer, properties={"element_type": "neuron"})[0]

    generators = []
    for neuron in neurons:
        generators.append(_set_input_current(neuron, current_dict, synaptic_strength, use_dc=use_dc))

    return generators


def convert_step_tuning(
        rf,
        neuron_tuning,
        tuning_discr_step
):
    """
    Convert the values from the receptive field to an activation according to the step function
    :param target_node: Node that receives input from the receptive field
    :param rf: The receptive field values
    :param neuron_tuning: Tuning class of the neuron
    :param tuning_discr_step: The margin of a single step that is within a class. If there are 4 classes the step size
    for every class is 255 / 4
    :param indices: The indices that correspond to the receptive field
    :param adj_mat: The adjacency / weight matrix
    :param min_target: Minimum id of the sensory neurons
    :return: Amplitudes
    """
    amplitude = np.zeros(rf.shape)

    mask = np.where(
        step_tuning_curve(
            rf,
            neuron_tuning,
            tuning_discr_step
        )
    )
    amplitude[mask] = 255.
    return amplitude


def convert_gauss_tuning(
        rf,
        neuron_tuning,
        tuning_discr_step
):
    """
    Convert the values from the receptive field to an activation according to the Gauss function
    :param target_node: Node that receives input from the receptive field
    :param rf: The receptive field values
    :param neuron_tuning: Tuning class of the neuron
    :param tuning_discr_step: The margin of a single step that is within a class. If there are 4 classes the step size
    for every class is 255 / 4
    :param indices: The indices that correspond to the receptive field
    :param adj_mat: The adjacency / weight matrix
    :param min_target: Minimum id of the sensory neurons
    :return: Amplitudes
    """
    amplitude = continuous_tuning_curve(rf, neuron_tuning, tuning_discr_step)
    return amplitude


def convert_linear_tuning(
        rf,
        neuron_tuning,
        tuning_discr_step
):
    """
    Convert the values from the receptive field to an activation according to a linear function
    :param target_node: Node that receives input from the receptive field
    :param rf: The receptive field values
    :param neuron_tuning: Tuning class of the neuron
    :param tuning_discr_step: The margin of a single step that is within a class. If there are 4 classes the step size
    for every class is 255 / 4
    :param indices: The indices that correspond to the receptive field
    :param adj_mat: The adjacency / weight matrix
    :param min_target: Minimum id of the sensory neurons
    :return: Amplitudes
    """
    amplitude, slope, intercept = linear_tuning(rf, neuron_tuning, tuning_discr_step)
    return amplitude


def create_rf(net, image_size=(50, 50)):
    # Follow the nest convention [columns, rows]
    mask_specs = {
        "lower_left": [-net.rf_size[0] // 2, -net.rf_size[1] // 2],
        "upper_right": [net.rf_size[0] // 2, net.rf_size[1] // 2]
    }
    net.ff_weight_mat = np.zeros((image_size[0] * image_size[1], net.num_sensory))
    index_values = np.arange(0, image_size[0] * image_size[1], 1).astype("int").reshape(image_size)
    min_id_target = min(net.torus_layer_nodes)

    rf_list = []
    for num, (target_node, rf_c) in enumerate(zip(net.torus_layer_nodes, net.rf_center_map)):
        upper_left = (np.asarray(rf_c) + np.asarray(mask_specs["lower_left"])).astype('int')
        lower_right = (np.asarray(rf_c) + np.asarray(mask_specs["upper_right"])).astype('int')

        upper_left = np.minimum(np.maximum(upper_left, 0), image_size[0])
        lower_right = np.minimum(np.maximum(lower_right, 0), image_size[0])
        indices = index_values[
                  upper_left[0]:lower_right[0],
                  upper_left[1]:lower_right[1]
                  ]

        rf_list.append((upper_left, lower_right[0] - upper_left[0], lower_right[1] - upper_left[1]))

        connections = np.random.binomial(1, net.p_rf, size=indices.size)
        indices = indices[connections.astype('bool').reshape(indices.shape)]

        # Set connections in matrix
        net.ff_weight_mat[indices.reshape(-1), target_node - min_id_target] = 1.

    return rf_list


def create_connections_rf(
        image,
        sampled_neurons,
        net,
        rf_list=None,
        plot_point=9,
):
    """
    Create receptive fields for sensory neurons and establishes connections
    :param image: Input image
    :param target_node_ids: Neurons in the target layer (i.e. the sheet with sensory neurons) to an input is injected
    :param rf_centers: The centers of the receptive fields
    :param tuning_vector: The map from neuron to stimulus tuning
    :param inh_neurons: IDs of inhibitory neurons
    :param rf_size: The size of a receptive field
    :param tuning_function: The tuning function of the sensory neurons
    :param synaptic_strength: Synaptic weight for the connections
    :param total_num_target: Total number of neurons. Note that if a sampling rate < 1 is used, the number of the
    target_node_ids is lower than the this value
    :param p_rf: Connection probability to the cells in the receptive field
    :param target_layer_size: Size of the square sheet with the sensory neurons
    :param max_spiking: Maximal spiking rate of a Poisson spike generator
    :param presentation_time: The time the image is presented to the network. Only possible with a Poisson spike train
    :param calc_error: If set to true, the reconstruction error is calculated based in the injected input,
    the reconstruction method is applied and the results is saved or displayed
    :param use_dc: If set to True a DC is injected, otherwise a Poisson spike train
    :param ff_factor: factor that scales the ff weight and hence must be the divisor for the rate
    :param plot_src_target: Flag for plotting
    :param save_plot: Flag for saving the plot. If plot_src_target is False this parameter is ignored
    :param save_prefix: Naming prefix for the saved plot. Is ignored if save_plot or plot is False
    :param non_changing_connections: If set to True, the function establishes the same conenctions if the other
    parameters remain unchanged
    :param plot_point: Number determining after how many established connections the plot is made
    :param retina_size: Size of the retina / input layer
    :param color_mask: Color mask of the functional map to plot receptive fields and sensory neurons properly
    :return: Adjacency matrix from receptors to sensory nodes
    """
    num_tuning_discr = int(net.tuning_vector.max() + 1)
    tuning_discr_step = 256. / float(num_tuning_discr)
    min_id_target = min(net.torus_layer_nodes)

    max_scale = net.rf_size[0] * net.rf_size[1] * 255.

    tuning_fun = None
    if net.tuning_function == TUNING_FUNCTION["step"]:
        tuning_fun = convert_step_tuning
    elif net.tuning_function == TUNING_FUNCTION["gauss"]:
        tuning_fun = convert_gauss_tuning
    elif net.tuning_function == TUNING_FUNCTION["linear"]:
        tuning_fun = convert_linear_tuning
    else:
        raise ValueError("The passed tuning function is not supported")

    counter = 0
    amplitudes = np.zeros(net.num_sensory)
    for num, target_node in enumerate(sampled_neurons):
        counter += 1

        if counter == plot_point:
            if net.plot_rf_relation:
                plot_rf(net, image, rf_list, counter=counter)

        # Choose connections
        connections = net.ff_weight_mat[:, num]

        rf = image.reshape(-1)[connections.astype('bool')]

        # Determine input based on RF
        if target_node not in net.torus_inh_nodes:
            amplitude = tuning_fun(
                rf,
                net.tuning_vector[target_node - min_id_target],
                tuning_discr_step
            )
        else:
            amplitude = rf

        if net.use_dc:
            inject = np.maximum((amplitude.sum() / max_scale) / net.ff_factor, 0.)
            current_dict = {"amplitude": inject}
        else:
            inject = np.maximum((net.max_spiking / net.ff_factor) * amplitude.sum() / max_scale, 0)
            current_dict = {"rate": inject, "stop": net.presentation_time}

        amplitudes[target_node - min_id_target] = inject
        # Create new input generator if they haven't been created yet. This is usually the case with dc
        if net.spike_gen is None:
            _set_input_current(
                target_node,
                current_dict,
                net.ff_weight,
                use_dc=net.use_dc
            )
        else:
            nest.SetStatus([net.spike_gen[target_node - min_id_target]], current_dict)

    recons = None
    if net.use_input_neurons:
        import modules.stimulusReconstruction as sr
        recons = sr.oblivious_stimulus_reconstruction(
            amplitudes,
            net.ff_weight_mat,
            net.tuning_vector
        )

        if net.plot_rf_relation:
            plot_reconstruction(
                image,
                recons,
                color_mask=net.color_map,
                save_plots=net.save_plots,
                save_prefix=net.save_prefix
            )

            plot_tuning_classes(
                tuning_fun,
                num_tuning_discr=num_tuning_discr,
                save_plot=net.save_plots,
                save_prefix=net.save_prefix
            )

    return recons

