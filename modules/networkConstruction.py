#!/usr/bin/python
from modules.thesisUtils import *
from modules.networkAnalysis import *

import warnings
import numpy as np
import scipy.interpolate as ip
import matplotlib.patches as patches
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import nest.topology as tp
import nest

# Define global constants
GLOBAL_CONNECTIVITY = 0.0123
R_MAX = 8.


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

    return (c_lr * float(layer_size)**2) / (num_patches * patchy_area)


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

    return c_lr / ((full_area - inner_area) / float(layer_size)**2)


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
        rest_pot=0.,
        threshold_pot=1e3,
        time_const=20.,
        capacitance=1e12,
        size_layer=R_MAX
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
    :return: neural layer
    """
    # Calculate positions
    positions = np.random.uniform(- size_layer / 2., size_layer / 2., size=(num_neurons, 2)).tolist()
    # Create dict for neural layer that is wrapped as torus to avoid boundary effects
    torus_dict = {
        "extent": [size_layer, size_layer],
        "positions": positions,
        "elements": neuron_type,
        "edge_wrap": True
    }
    if neuron_type is "iaf_psc_delta" or neuron_type is "iaf_psc_alpha":
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
    spikedetector = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
    multimeter = nest.Create("multimeter", params={"withtime": True, "record_from": ["V_m"]})
    nest.Connect(sensory_nodes, spikedetector)
    nest.Connect(multimeter, sensory_nodes)
    return torus_layer, spikedetector, multimeter


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


def create_local_circular_connections(
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
        r_loc=0.5,
        p_loc=0.7,
        num_patches=3,
        p_p=None
):
    """
    Create random long range patchy connections. To every neuron a single link is established instead of
    taking axonal morphology into account.
    :param layer: Layer in which the connections should be established
    :param r_loc: Radius for local connections
    :param p_loc: Probability for local connections
    :param num_patches: Number of patches that should be created
    :param p_p: Probability to establish long-range patchy connections. If none, prob. is calculated according to Voges
                paper
    :return Nodes of the layer for debugging purposes (plotting)
    """
    # Calculate the parameters for the patches
    layer_size = nest.GetStatus(layer, "topology")[0]["extent"][0]
    r_p = r_loc / 2.
    min_distance = r_loc + r_p
    max_distance = layer_size / 2. - r_loc
    if p_p is None:
        p_p = get_lr_connection_probability_patches(r_loc, p_loc, r_p, num_patches=num_patches, layer_size=layer_size)

    # Iterate through all neurons, as all neurons have random patches
    nodes = nest.GetNodes(layer)[0]
    for neuron in nodes:
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
        nest.Connect([neuron], patches, connect_dict)

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
    :param p_loc: Probility of local connections
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
        neuron_to_tuning_map,
        tuning_to_neuron_map,
        r_loc=0.5,
        connect_dict=None,
        plot=False,
        save_plot=False,
        color_mask=None
):
    """
    Create local connections only to neurons with same stimulus feature preference
    :param layer: Layer with neurons
    :param neuron_to_tuning_map: Dictionary mapping from neurons to their respective tuning preference
    :param tuning_to_neuron_map: Dictionary mapping from tuning preference to all neurons with that preference
    :param r_loc: Radius within which local connections can be established
    :param connect_dict: Conenction dictionary specifying the connection parameter
    :param plot: Flag for plotting
    :param save_plot: Flag for saving the plot. If plot is False this parameter is ignored
    :param color_mask: Color/orientation map for plotting. If plot is False this parameter is ignored
    """
    if connect_dict is None:
        connect_dict = {
            "rule": "pairwise_bernoulli",
            "p": 0.7
        }

    node_ids = nest.GetNodes(layer)[0]
    for node in node_ids:
        stimulus_tuning = neuron_to_tuning_map[node]
        similar_tuned_neurons = tuning_to_neuron_map[stimulus_tuning]
        same_stimulus_area = list(filter(lambda l: node in l, similar_tuned_neurons))[0]
        same_stimulus_area = list(filter(lambda n: n != node, same_stimulus_area))
        connect_partners = [
            connect_p for connect_p in same_stimulus_area
            if tp.Distance([connect_p], [node])[0] < r_loc
        ]
        nest.Connect([node], connect_partners, connect_dict)

        # Plot first one
        if node - min(node_ids) == 0:
            if plot:
                # Assume that layer is a square
                layer_size = nest.GetStatus(layer, "topology")[0]["extent"][0]
                connect = nest.GetConnections([node])
                targets = nest.GetStatus(connect, "target")
                local_targets = [t for t in targets if t in list(connect_partners)]
                plot_connections(
                    [node],
                    local_targets,
                    save_plot=save_plot,
                    plot_name="local_connections.png",
                    color_mask=color_mask,
                    layer_size=layer_size
                )


def create_stimulus_based_patches_random(
        layer,
        neuron_to_tuning_map,
        tuning_to_neuron_map,
        num_patches=2,
        r_loc=0.5,
        p_loc=0.7,
        p_p=0.7,
        r_p=None,
        connect_dict=None,
        plot=False,
        save_plot=False,
        color_mask=None
):
    """
    Create patchy connections based on tuning preference
    :param layer: Layer with neurons
    :param neuron_to_tuning_map: Dictionary mapping from neurons to their respective tuning preference
    :param tuning_to_neuron_map: Dictionary mapping from tuning preference to all neurons with that preference
    :param num_patches: Number of patches that are created
    :param r_loc: Radius for local connections
    :param p_loc: Probability for local connections. If p_p or the connect dict is not None this parameter is ignored
    :param p_p: Probability for patchy connections
    :param r_p: Patchy radius. If None it is half of the local radius
    :param connect_dict: Dictionary specifying the connections that are to establish
    :param plot: Flag for plotting
    :param save_plot: Flag for saving the plot. If plot is set to False this parameter is ignored
    :param color_mask: Color/orientation map. If plot is set to False this parameter is ignored
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
    max_distance = size_layer / 2. - r_p
    if p_p is None and connect_dict is None:
        p_p = get_lr_connection_probability_patches(r_loc, p_loc, r_p, num_patches=num_patches, layer_size=size_layer)
    node_ids = nest.GetNodes(layer)[0]
    mask_specs = {"radius": r_p}

    for neuron in node_ids:
        stimulus_tuning = neuron_to_tuning_map[neuron]
        same_tuning_nodes = tuning_to_neuron_map[stimulus_tuning]
        if len(same_tuning_nodes) > 1:
            same_tuning_nodes = [area for area in same_tuning_nodes if neuron not in area]
        patch_center_nodes = []
        while len(patch_center_nodes) < num_patches:
            area_idx = np.random.choice(len(same_tuning_nodes))
            area = same_tuning_nodes[area_idx]
            patch_center = np.random.choice(area)
            if min_distance <= tp.Distance([neuron], [int(patch_center)])[0] < max_distance:
                patch_center_nodes.append(tp.GetPosition([int(patch_center)])[0])

        lr_patches = tuple()
        for neuron_anchor in patch_center_nodes:
            lr_patches += tp.SelectNodesByMask(
                layer,
                neuron_anchor,
                mask_obj=tp.CreateMask("circular", specs=mask_specs)
            )
            stimulus_tuning = neuron_to_tuning_map[neuron]
            lr_patches = tuple(filter(lambda n: neuron_to_tuning_map[n] == stimulus_tuning, lr_patches))

        # Define connection
        if connect_dict is None:
            connect_dict = {
                "rule": "pairwise_bernoulli",
                "p": p_p
            }
        nest.Connect([neuron], lr_patches, connect_dict)

        # Plot only first neuron
        if neuron - min(node_ids) == 4:
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
                    plot_name="lr-patchy-connections.png"
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
    :param adj_mat: Adjacency matrix for the connections
    :param cap_s: Weight for the connections
    :param divide_by_num_connect: Flag for dividing the weight through the number of established connections
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
        organise_on_grid=False
):
    """
    Create direct current generator to simulate input stimulus. The pixel values of the image are transformed
    to an integer value representing the intensity in Ampere A
    :param input_stimulus: Grayscale input stimulus with integer values between 0 and 256
    :param organise_on_grid: Flag for organising current generators on a grid to represent receptors
    :return: Tuple with ids for the dc generator devices
    """
    assert np.all(input_stimulus < 256) and np.all(input_stimulus >= 0)

    num_receptors = input_stimulus.size
    # Multiply value with 1e12, as the generator expects values in pA
    current_dict = [{"amplitude": float(amplitude * 1e12)} for amplitude in input_stimulus.reshape(-1)]
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
        num_stimulus_discr=4,
        resolution=(10, 10),
        spacing=0.1,
        plot=False,
        save_plot=False,
        plot_name=None
):
    size_layer = nest.GetStatus(layer, "topology")[0]["extent"][0]
    nodes = nest.GetNodes(layer)[0]
    min_idx = min(nodes)

    tuning_to_neuron_map = {stimulus: [] for stimulus in range(num_stimulus_discr)}
    neuron_to_tuning_map = {}
    tuning_weight_vector = np.zeros(len(nodes))

    grid_nodes_range = np.arange(0, size_layer, spacing)
    stimulus_grid_range_x = np.arange(0, resolution[0])
    stimulus_grid_range_y = np.arange(0, resolution[1])
    V = np.random.rand(stimulus_grid_range_x.size, stimulus_grid_range_y.size)

    ipol = ip.RectBivariateSpline(stimulus_grid_range_x, stimulus_grid_range_y, V)
    color_map = ipol(grid_nodes_range, grid_nodes_range)
    color_map = np.round(color_map * (num_stimulus_discr - 1)).astype('int')

    if plot:
        plt.imshow(
            color_map,
            origin=(stimulus_grid_range_x.size//2, stimulus_grid_range_y.size//2),
            extent=(-size_layer/2., size_layer/2., -size_layer/2., size_layer/2.),
            cmap='tab10',
            alpha=0.4
        )

    for n in nodes:
        p = tp.GetPosition([n])[0]
        # Grid positions
        y_grid = int(((size_layer/2.) + p[0]) / spacing)
        x_grid = int(((size_layer/2.) + p[1]) / spacing)

        stim_class = color_map[x_grid, y_grid]

        tuning_to_neuron_map[stim_class].append(n)
        neuron_to_tuning_map[n] = stim_class
        tuning_weight_vector[n - min_idx] = (stim_class + 1) / num_stimulus_discr

        if plot:
            plt.plot(
                p[0],
                p[1],
                marker='o',
                markerfacecolor=list(mcolors.TABLEAU_COLORS.items())[stim_class][0],
                markeredgewidth=0
            )

    if plot:
        if not save_plot:
            plt.show()
        else:
            curr_dir = os.getcwd()
            if plot_name is None:
                plot_name = "stimulus_tuning_map.png"
            plt.savefig(curr_dir + "/figures/" + plot_name)
    return tuning_to_neuron_map, neuron_to_tuning_map, tuning_weight_vector, color_map


def create_stimulus_tuning_map(
        layer,
        num_stimulus_discr=4,
        stimulus_per_row=2,
        plot=False,
        save_plot=False,
        plot_name=None
):
    """
    Create the stimulus tuning map for neurons
    :param layer: Layer of neurons
    :param num_stimulus_discr: Number of stimuli feature classes that are to discriminate
    :param stimulus_per_row: Number of how often a stimulus should be represented per row
    :param plot: Flag for plotting
    :param save_plot: Flag for saving the plot. If plot is set to False this parameter is ignored
    :param plot_name: Name of the saved plot file. If plot is set to False this parameter is ignored
    :return: Map from stimulus feature class to neuron, map form neuron to stimulus feature class, vector with weights
            for their respective class, e.g. class of neurons / # classes
    """
    size_layer = nest.GetStatus(layer, "topology")[0]["extent"][0]
    # Assume quadratic layer size
    box_size = size_layer / float(stimulus_per_row * num_stimulus_discr)
    sublayer_anchors, box_mask_dict = create_distinct_sublayer_boxes(size_boxes=box_size, size_layer=size_layer)
    tuning_to_neuron_map = {stimulus: [] for stimulus in range(num_stimulus_discr)}
    neuron_to_tuning_map = {}
    nodes = nest.GetNodes(layer)[0]
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
        tuning_to_neuron_map[stimulus_tuning].append(stimulus_area)
        for neuron in stimulus_area:
            tuning_weight_vector[neuron - min_idx] = stimulus_tuning / (num_stimulus_discr - 1)
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
                alpha=0.2
            )
            plt.gca().add_patch(area_rect)
            plt.plot()

    if plot:
        if not save_plot:
            plt.show()
            plt.close()
        else:
            curr_dir = os.getcwd()
            if plot_name is None:
                plot_name = "stimulus_tuning_map.png"
            plt.savefig(curr_dir + "/figures/" + plot_name)

    return tuning_to_neuron_map, neuron_to_tuning_map, tuning_weight_vector, color_map


def check_in_stimulus_tuning(
        input_stimulus,
        stimulus_tuning,
        tuning_discr_steps
):
    """
    Function to check whether a certain neuron reacts on a certain stimulus (e.g. stimulus is within the stimulus
    feature class)
    :param input_stimulus: The input stimulus
    :param stimulus_tuning: The tuning of the neuron (identifier for the class)
    :param tuning_discr_steps: The number of discriminated stimulus features
    :return: True if neuron reacts, False otherwise
    """
    return stimulus_tuning * tuning_discr_steps <= input_stimulus / 1e12 < (stimulus_tuning + 1) * tuning_discr_steps


def create_connections_rf(
        src_layer,
        target_layer,
        rf_centers,
        neuron_to_tuning_map,
        rf_size=(5, 5),
        connect_dict=None,
        synaptic_strength=1.,
        ignore_weights=True,
        plot_src_target=False,
        save_plot=False,
        plot_point=10,
        retina_size=(100, 100)
):
    """
    Create receptive fields for sensory neurons and establishes connections
    :param src_layer: Source layer, i.e. the retina layer with receptors
    :param target_layer: Target layer, i.e. the layer with sensory neurons
    :param rf_centers: The centers of the receptive fields
    :param neuron_to_tuning_map: The map from neuron to stimulus tuning
    :param rf_size: The size of a receptive field
    :param connect_dict: Dictionary defining the connection values. If None default values are used
    :param synaptic_strength: Synaptic weight for the connections
    :param ignore_weights: Flag for creating the adjacency matrix. If set to True, all connections are considered in
                             the matrix, whereas if it is set to False, only connections with non-zero weight
                             are considered
    :param plot_src_target: Flag for plotting
    :param save_plot: Flag for saving the plot. If plot_src_target is False this parameter is ignored
    :param plot_point: Number determining after how many established connections the plot is made
    :param retina_size: Size of the retina / input layer
    :return: Adjacency matrix from receptors to sensory nodes
    """
    target_node_ids = nest.GetNodes(target_layer)[0]
    src_node_ids = nest.GetNodes(src_layer)[0]

    # Follow the nest convention [columns, rows]
    mask_specs = {
        "lower_left": [-rf_size[1] / 2., -rf_size[0] / 2.],
        "upper_right": [rf_size[1] / 2., rf_size[0] / 2.]
    }
    num_tuning_discr = max(neuron_to_tuning_map.values()) + 1
    tuning_discr_step = 255 / float(num_tuning_discr)
    min_id_target = min(target_node_ids)
    min_id_src = min(src_node_ids)
    adj_mat = np.zeros((len(src_node_ids), len(target_node_ids)), dtype='uint8')

    if connect_dict is None:
        connect_dict = {
            "rule": "pairwise_bernoulli",
            "p": 0.7
        }

    counter = 0
    rf_list = []
    for target_node, rf_center in zip(target_node_ids, rf_centers):
        src_nodes = tp.SelectNodesByMask(src_layer, rf_center, mask_obj=tp.CreateMask("rectangular", specs=mask_specs))
        src_pos = tp.GetPosition(src_nodes)
        rf_list.append(list(src_pos))

        nest.Connect(src_nodes, [target_node], connect_dict)
        connections = nest.GetConnections(source=src_nodes, target=[target_node])
        connected_src_nodes = nest.GetStatus(connections, "source")
        synapse_declaration = [
            {"weight": 0. if not check_in_stimulus_tuning(
                nest.GetStatus([receptor], "amplitude")[0],
                neuron_to_tuning_map[target_node],
                tuning_discr_step) else synaptic_strength
             } for receptor in connected_src_nodes]

        if plot_src_target:
            if counter == plot_point:
                target_layer_size = nest.GetStatus(target_layer, "topology")[0]["extent"][0]

                fig, ax = plt.subplots(1, 2, sharex='none', sharey='none')
                ax[0].axis((-retina_size[1]/2., retina_size[1]/2., -retina_size[0]/2., retina_size[0]/2.))
                for rf in rf_list:
                    # De-zip to get x and y values separately
                    x_pixel, y_pixel = zip(*rf)
                    ax[0].plot(x_pixel, y_pixel, '.')
                ax[0].set_xlabel("Retina tissue in X")
                ax[0].set_ylabel("Retina tissue in Y")

                target_positions = tp.GetPosition(list(target_node_ids[max(counter - plot_point, 0): counter + 1]))
                # Iterate over values to have matching colors
                for x, y in target_positions:
                    ax[1].plot(x, y, 'o')
                ax[1].set_xlabel("V1 tissue in X")
                ax[1].set_ylabel("V1 tissue in Y")
                ax[1].set_xlim([-target_layer_size / 2., target_layer_size / 2.])
                ax[1].set_ylim([-target_layer_size / 2., target_layer_size / 2.])

                if save_plot:
                    curr_dir = os.getcwd()
                    plt.savefig(curr_dir + "/figures/receptive_fields.png")
                    plt.close()
                else:
                    plt.show()

        counter += 1

        nest.SetStatus(connections, synapse_declaration)
        adj_mat = set_values_in_adjacency_matrix(
            connections,
            adj_mat,
            min_id_src,
            min_id_target,
            ignore_weights=ignore_weights
        )

    nest.SetStatus(src_node_ids, {"amplitude": 255e12})
    return adj_mat
