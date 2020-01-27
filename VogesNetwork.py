#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import nest.topology as tp

# Define global constants
GLOBAL_CONNECTIVITY = 0.0123
R_MAX = 8.


def to_coordinates(angle, distance):
    """
    Convert distance and angle to coordinates
    :param angle: angle
    :param distance: distance
    :return: coordinates [x, y]
    """
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
    inner_area = np.pi * r_loc ** 2
    c_loc = p_loc * inner_area / float(R_MAX) ** 2
    return c_loc, inner_area


def get_lr_connection_probability_patches(
        r_loc,
        p_loc,
        r_p
):
    """
    Calculate the connection probability of long range patchy connections
    :param r_loc: radius of local connections
    :param p_loc: probability of establishing local connections
    :param r_p: patchy radius
    :return: long range patchy connection probability
    """
    c_loc, _ = get_local_connectivity(r_loc, p_loc)
    patchy_area = np.pi * r_p ** 2
    c_lr = GLOBAL_CONNECTIVITY - c_loc

    return c_lr / (patchy_area / float(R_MAX) ** 2)


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
    """
    mask_dict = {
        "doughnut": {
            "inner_radius": r_loc,
            "outer_radius": R_MAX / 2.
        }
    }

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


def create_random_patches(
        layer,
        r_loc=0.5,
        p_loc=0.7,
        num_patches=3,
        allow_multapses=False
):
    """
    Create random long range patchy connections. To every neuron a single link is established instead of
    taking axonal morphology into account.
    :param layer: Layer in which the connections should be established
    :param r_loc: Radius for local connections
    :param p_loc: Probability for local connections
    :param num_patches: Number of patches that should be created
    :param allow_multapses: Flag to allow multiple connections betwenn neurons
    """
    # Calculate radial distance and the respective coordinates for patches
    r_p = r_loc / 2.
    min_distance = r_loc + r_p
    max_distance = R_MAX / 2. - r_loc
    radial_distance = np.random.uniform(min_distance, max_distance, size=num_patches).tolist()
    radial_angle = np.random.uniform(0., 359., size=num_patches).tolist()

    p_p = get_lr_connection_probability_patches(r_loc, p_loc, r_p)

    for angle, distance in zip(radial_angle, radial_distance):
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


def main(
        plot_torus=True,
        plot_target=True
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
    create_random_patches(torus_layer)
    if plot_target:
        tp.PlotTargets([2], torus_layer)
        plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    main(plot_torus=False)


