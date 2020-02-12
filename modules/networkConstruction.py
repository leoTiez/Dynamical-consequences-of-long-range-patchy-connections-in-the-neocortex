#!/usr/bin/python
from modules.thesisUtils import *
from modules.networkAnalysis import *

import warnings
import numpy as np
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import nest.topology as tp
import nest

# Define global constants
GLOBAL_CONNECTIVITY = 0.0123
R_MAX = 8.


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


def create_distinct_sublayer_boxes(size_boxes, size_layer=R_MAX):
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
        size_layer = R_MAX
):
    """
    Create a layer wrapped a torus to avoid boundary conditions. Neurons are placed uniformly
    :param num_neurons: Number of neurons in the layer
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
        neuron_type="iaf_psc_alpha"
):
    """
    Create a layer wrapped a torus to avoid boundary conditions. Neurons are placed on a grid with fluctuations
    :param num_neurons: Number of neurons in layer
    :param jitter: amount of jitter
    :return: layer
    """
    # Create coordinates of neurons
    mod_size = R_MAX - jitter*2
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


def create_location_based_patches(
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
    sublayer_anchors, box_mask_dict = create_distinct_sublayer_boxes(size_boxes)

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

    return create_location_based_patches(
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

    return create_location_based_patches(
        layer=layer,
        r_loc=r_loc,
        p_loc=p_loc,
        size_boxes=size_boxes,
        num_patches=num_patches,
        num_shared_patches=num_shared_patches,
        num_patches_replaced=num_patches_replaced,
        is_partially_overlapping=True

    )


def create_stimulus_based_local_connections(
        layer,
        neuron_to_tuning_map,
        tuning_to_neuron_map,
        r_loc=0.5,
        connect_dict=None
):
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


def create_stimulus_based_patches_random(
        layer,
        neuron_to_tuning_map,
        tuning_to_neuron_map,
        num_patches=2,
        r_loc=0.5,
        p_loc=0.7,
        p_p=0.7,
        connect_dict=None,
        size_layer=R_MAX,
):
    # Calculate parameters for the patches
    if connect_dict is None and p_p is None:
        warnings.warn("Connect dict not specified.\n"
                      "Computed probability based on Voges does not necessarily return in all cases a valid value.\n"
                      "Recommended values are p_loc >= 0.5 and r_loc > 0.5")

    r_p = r_loc / 2.
    min_distance = r_loc + r_p
    max_distance = size_layer / 2. - r_loc
    if p_p is None:
        p_p = get_lr_connection_probability_patches(r_loc, p_loc, r_p, num_patches=num_patches)
    node_ids = nest.GetNodes(layer)[0]
    mask_specs = {"radius": r_p}

    for neuron in node_ids:
        stimulus_tuning = neuron_to_tuning_map[neuron]
        same_tuning_nodes = tuning_to_neuron_map[stimulus_tuning]
        # same_tuning_nodes = [area for area in same_tuning_nodes if neuron not in area]
        patch_center_nodes = []
        while len(patch_center_nodes) < num_patches:
            area_idx = np.random.choice(len(same_tuning_nodes))
            area = same_tuning_nodes[area_idx]
            patch_center = np.random.choice(area)
            if min_distance <= tp.Distance([neuron], [int(patch_center)])[0] < max_distance:
                patch_center_nodes.append(tp.GetPosition([int(patch_center)])[0])

        patches = tuple()
        for neuron_anchor in patch_center_nodes:
            patches += tp.SelectNodesByMask(
                layer,
                neuron_anchor,
                mask_obj=tp.CreateMask("circular", specs=mask_specs)
            )
        # Define connection
        if connect_dict is None:
            connect_dict = {
                "rule": "pairwise_bernoulli",
                "p": p_p
            }
        nest.Connect([neuron], patches, connect_dict)


def set_synaptic_strenght(
        nodes,
        adj_mat,
        cap_s=1.
):
    num_connec = adj_mat.sum()
    if num_connec == 0:
        return
    connect = nest.GetConnections(source=nodes)
    connect = [c for c in connect if nest.GetStatus([c], "target")[0] in nodes]
    nest.SetStatus(connect, {"weight": cap_s/float(num_connec)})


def create_input_current_generator(
        input_stimulus,
        organise_on_grid=False
):
    """
    Create direct current generator to simulate input stimulus. The pixel values of the image are transformed
    to an integer value representing the intensity in Ampere A
    :param input_stimulus: Grayscale input stimulus with integer values between 0 and 256
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


def create_stimulus_tuning_map(
        layer,
        num_stimulus_discr=4,
        size_layer=R_MAX,
        plot=False,
        save_plot=False,
        plot_name=None
):
    # Assume quadratic layer size
    box_size = size_layer // num_stimulus_discr
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
            "color":color
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
        plot_freq=10,
        retina_size=(100, 100)
):
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
            if counter % plot_freq == 0:
                fig, ax = plt.subplots(1, 2)
                ax[0].axis((-retina_size[1]/2., retina_size[1]/2., -retina_size[0]/2., retina_size[0]/2.))
                for rf in rf_list:
                    # De-zip to get x and y values separately
                    x_pixel, y_pixel = zip(*rf)
                    ax[0].plot(x_pixel, y_pixel, '.')
                ax[0].set_xlabel("Retina tissue in X")
                ax[0].set_ylabel("Retina tissue in Y")

                ax[1].axis((-R_MAX / 2., R_MAX / 2., -R_MAX / 2., R_MAX / 2.))
                target_positions = tp.GetPosition(list(target_node_ids[max(counter - plot_freq, 0): counter + 1]))
                # Iterate over values to have matching colors
                for x, y in target_positions:
                    ax[1].plot(x, y, 'o')
                ax[1].set_xlabel("V1 tissue in X")
                ax[1].set_ylabel("V1 tissue in Y")
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
