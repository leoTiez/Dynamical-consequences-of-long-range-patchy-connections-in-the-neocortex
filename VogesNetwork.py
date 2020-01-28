#!/usr/bin/python3

# ####################################################################################################################
# This Python script simulates the networks with long-range connections that are established according to different
# parameters, as described in the paper by Voges et al.:
#
# Voges, N., Guijarro, C., Aertsen, A. & Rotter, S.
# Models of cortical networks with long-range patchy projections.
# Journal of Computational Neuroscience 28, 137–154 (2010).
# DOI: 10.1007/s10827-009-0193-z
# ####################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import nest.topology as tp
import nest
# Define global constants
GLOBAL_CONNECTIVITY = 0.0123
R_MAX = 8.


def degree_to_rad(deg):
    """
    Convert degree to radians
    :param deg: value in degree
    :return: value in radians
    """
    return deg * np.pi / 180.


def to_coordinates(angle, distance):
    """
    Convert distance and angle to coordinates
    :param angle: angle
    :param distance: distance
    :return: coordinates [x, y]
    """
    angle = degree_to_rad(angle)
    return [distance * np.cos(angle), distance * np.sin(angle)]


def get_local_connectivity(
        r_loc,
        p_loc
):
    """
    Calculate local connectivity
    :param r_loc: radius of local connections
    :param p_loc: probability of establishing local connections
    :return: local connectivity
    """
    inner_area = np.pi * r_loc**2
    c_loc = p_loc * inner_area / float(R_MAX)**2
    return c_loc, inner_area


def get_lr_connection_probability_patches(
        r_loc,
        p_loc,
        r_p,
        num_patches=3
):
    """
    Calculate the connection probability of long range patchy connections
    :param r_loc: radius of local connections
    :param p_loc: probability of establishing local connections
    :param r_p: patchy radius
    :param num_patches: Total number of patches
    :return: long range patchy connection probability
    """
    c_loc, _ = get_local_connectivity(r_loc, p_loc)
    patchy_area = np.pi * r_p**2
    c_lr = GLOBAL_CONNECTIVITY - c_loc

    return (c_lr * float(R_MAX)**2) / (num_patches * patchy_area)


def get_lr_connection_probability_np(
        r_loc,
        p_loc
):
    """
    Calculate long range connectivity probability according to Voges Paper
    :param r_loc: local radius
    :param p_loc: local connectivity probability
    :return: long range connectivity probability
    """

    full_area = np.pi * R_MAX**2
    # Calculate local connectivity
    c_loc, inner_area = get_local_connectivity(r_loc, p_loc)
    # Calculate long range connectivity
    c_lr = GLOBAL_CONNECTIVITY - c_loc

    return c_lr / ((full_area - inner_area) / float(R_MAX)**2)


def create_torus_layer_uniform(
        num_neurons=3600
):
    """
    Create a layer wrapped a torus to avoid boundary conditions. Neurons are placed uniformly
    :param num_neurons: Number of neurons in the layer
    :return: neural layer
    """
    # Calculate positions
    positions = np.random.uniform(- R_MAX / 2., R_MAX / 2., size=(num_neurons, 2)).tolist()
    # Create dict for neural layer that is wrapped as torus to avoid boundary effects
    torus_dict = {
        "extent": [R_MAX, R_MAX],
        "positions": positions,
        "elements": "iaf_psc_alpha",
        "edge_wrap": True
    }
    # Create layer
    torus_layer = tp.CreateLayer(torus_dict)
    return torus_layer


def create_torus_layer_with_jitter(
        num_neurons=3600,
        jitter=0.03
):
    """
    Create a layer wrapped a torus to avoid boundary conditions. Neurons are placed on a grid with fluctuations
    :param num_neurons: Number of neurons in layer
    :param jitter: amount of jitter
    :return: layer
    """
    # Create coordinates of neurons
    mod_size = R_MAX- jitter*2
    step_size = mod_size / float(np.sqrt(num_neurons))
    coordinate_scale = np.arange(-mod_size / 2., mod_size / 2., step_size)
    grid = [[x, y] for y in coordinate_scale for x in coordinate_scale]
    positions = [[pos[0] + np.random.uniform(-jitter, jitter),
                 pos[1] + np.random.uniform(-jitter, jitter)]
                 for pos in grid]

    # Create dict for neural layer that is wrapped as torus to avoid boundary effects
    torus_dict = {
        "extent": [R_MAX, R_MAX],
        "positions": positions,
        "elements": "iaf_psc_alpha",
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
        allow_multapses=False
):
    """
    Create local connections with in a circular radius
    :param layer: The layer where the local connections should be established
    :param r_loc: radius for local connections
    :param p_loc: probability of establishing local connections
    :param allow_autapses: Flag to allow self-connections
    :param allow_multapses: Flag to allow multiple connections between neurons
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


def create_distant_np_connections(
        layer,
        r_loc=0.5,
        p_loc=0.7,
        allow_multapses=False
):
    """
    Create long distance connections without any patches
    :param layer: Layer in which the connections should be established
    :param r_loc: radius for local connections needed to calculate the long range connection probability
    :param p_loc: probability for local connections needed to calculate the long range connection probability
    :param allow_multapses: allow multiple connections between neurons
    :return Neurons of the layer for debugging (plotting)
    """
    # Mask for area to which long-range connections can be established
    mask_dict = {
        "doughnut": {
            "inner_radius": r_loc,
            "outer_radius": R_MAX / 2.
        }
    }

    # Get possibility to establish a single long-range connection
    p_lr = get_lr_connection_probability_np(r_loc, p_loc)

    connection_dict = {
        "connection_type": "divergent",
        "mask": mask_dict,
        "kernel": p_lr,
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
):
    """
    Create random long range patchy connections. To every neuron a single link is established instead of
    taking axonal morphology into account.
    :param layer: Layer in which the connections should be established
    :param r_loc: Radius for local connections
    :param p_loc: Probability for local connections
    :param num_patches: Number of patches that should be created
    :return Nodes of the layer for debugging purposes (plotting)
    """
    # Calculate the parameters for the patches
    r_p = r_loc / 2.
    min_distance = r_loc + r_p
    max_distance = R_MAX / 2. - r_loc
    p_p = get_lr_connection_probability_patches(r_loc, p_loc, r_p, num_patches=num_patches)

    # Iterate through all neurons, as all neurons have random patches
    nodes = nest.GetNodes(layer)[0]
    for neuron in nodes:
        # Calculate radial distance and the respective coordinates for patches
        radial_distance = np.random.uniform(min_distance, max_distance, size=num_patches).tolist()
        radial_angle = np.random.uniform(0., 359., size=num_patches).tolist()

        # Calculate patch region
        mask_specs = {"radius": r_p}
        anchors = [to_coordinates(distance, angle) for angle, distance in zip(radial_angle, radial_distance)]
        patches = tuple()
        for anchor in anchors:
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
        allow_multapses=False
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
    :return: Neurons of the layer for debugging (plotting)
    """
    # Calculate parameters for pathces
    r_p = r_loc / 2.
    p_p = get_lr_connection_probability_patches(r_loc, p_loc, r_p)

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


def create_localtion_based_patches(
        layer,
        r_loc=0.5,
        p_loc=0.7,
        size_boxes=0.5,
        num_patches=3,
        num_shared_patches=6,
        num_patches_replaced=3,
        is_partially_overlapping=False
):
    """
    Function to establish patchy connections for neurons that have a location based relationship, such that
    they are in the same sublayer / box
    :param layer: Layer in which the connections should be established
    :param r_loc: Radius of local connections
    :param p_loc: Probaility of local connections
    :param size_boxes: Size of a sublayer in which neurons share patches. The R_MAX / size_boxes
    should be an integer value
    :param num_patches: Number of patches per neuron
    :param num_shared_patches: Number of patches per box that are shared between the neurons
    :param num_patches_replaced: Number of patches that are replaced in x-direction (for partially overlapping patches)
    :param is_partially_overlapping: Flag for partially overlapping patches
    :return: Sublayer at size_boxes/2 for debugging purposes (plotting)
    """
    # Calculate parameters for the patches
    r_p = r_loc / 2.
    min_distance = r_loc + r_p
    max_distance = R_MAX / 2. - r_loc
    p_p = get_lr_connection_probability_patches(r_loc, p_loc, r_p, num_patches=num_patches)

    # Create sublayer boxes that share same patches
    sublayer_anchors = [[x*size_boxes + size_boxes/2., y*size_boxes + size_boxes/2.]
               for y in range(-int(R_MAX/(2.*float(size_boxes))), int(R_MAX/(2.*float(size_boxes))))
               for x in range(-int(R_MAX/(2.*float(size_boxes))), int(R_MAX/(2.*float(size_boxes))))
               ]
    box_mask_dict = {"lower_left": [-size_boxes/2., -size_boxes/2.], "upper_right": [size_boxes/2., size_boxes/2.]}

    last_y_anchor = -R_MAX - 1
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
        patches_anchors = [to_coordinates(distance, angle) for angle, distance in zip(radial_angle, radial_distance)]

        # Iterate through all neurons, as patches are chosen for each neuron independently
        for neuron in sub_layer:
            neuron_patch_anchors = np.asarray(patches_anchors)[
                np.random.choice(len(patches_anchors), size=num_patches, replace=False)
            ]
            patches = tuple()
            for neuron_anchor in neuron_patch_anchors.tolist():
                patches += tp.SelectNodesByMask(
                    layer,
                    neuron_anchor,
                    mask_obj=tp.CreateMask("circular", specs=mask_specs)
                )
            # Define connection
            connect_dict = {
                "rule": "pairwise_bernoulli",
                "p": p_p
            }
            nest.Connect([neuron], patches, connect_dict)

    # Return last sublayer for debugging
    return debug_sub_layer


def create_shared_patches(
        layer,
        r_loc=0.5,
        p_loc=0.7,
        size_boxes=0.5,
        num_patches=3,
        num_shared_patches=6,
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
    :return: Neurons of sublayer anchored at size_boxes/2 for debugging (plotting)
    """
    # Number of patches per neuron must be lower or equal to the number of patches per box, as the patches of a neuron
    # are a subset of the patches of the sublayer
    assert num_patches <= num_shared_patches

    return create_localtion_based_patches(
        layer=layer,
        r_loc=r_loc,
        p_loc=p_loc,
        size_boxes=size_boxes,
        num_patches=num_patches,
        num_shared_patches=num_shared_patches,
        is_partially_overlapping=False
    )


def create_partially_overlapping_patches(
        layer,
        r_loc=0.5,
        p_loc=0.7,
        size_boxes=0.5,
        num_patches=3,
        num_shared_patches=6,
        num_patches_replaced=3,

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
    :return: Neurons of the sublayer anchored at box_size/2 for debugging (plotting)
    """
    assert num_patches_replaced <= num_shared_patches
    assert num_patches <= num_shared_patches

    return create_localtion_based_patches(
        layer=layer,
        r_loc=r_loc,
        p_loc=p_loc,
        size_boxes=size_boxes,
        num_patches=num_patches,
        num_shared_patches=num_shared_patches,
        num_patches_replaced=num_patches_replaced,
        is_partially_overlapping=True

    )


def main(
        plot_torus=True,
        plot_target=True,
        num_plot_tagets=3
):
    """
    Main function running the test routines
    :param plot_torus: Flag to plot neural layer
    :param plot_target: Flag to plot targets to control established connections
    """
    torus_layer = create_torus_layer_uniform()
    if plot_torus:
        fig, _ = plt.subplots()
        tp.PlotLayer(torus_layer, fig)
        plt.show()

    create_local_circular_connections(torus_layer)

    debug_layer = create_partially_overlapping_patches(torus_layer)
    if plot_target:
        choice = np.random.choice(np.asarray(debug_layer), num_plot_tagets, replace=False)
        for c in choice:
            tp.PlotTargets([int(c)], torus_layer)
        plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    main(plot_torus=False)


