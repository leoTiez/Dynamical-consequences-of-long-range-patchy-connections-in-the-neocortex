#!/usr/bin/python3
import os
from pathlib import Path
import numpy as np
import pandas as pd
import ast
from scipy.spatial import KDTree
import modules.networkConstruction as nc
from modules.thesisUtils import print_msg

import nest


def convert_string_to_list(tuple_string, dtype=int):
    """
    Convert a tuple string to a list of values. Must be one-dimensional
    :param tuple_string: The string with the tuple
    :param dtype: Type to which the values are converted
    :return: List with values
    """
    return [dtype(s) for s in tuple_string.strip("(").strip(")").split(", ")]


def save_net(net, network_name, feature_folder, path="", use_cwd=True):
    """
    Save the neurons and connections of the network to be loaded quicker later
    :param net: The network object
    :param network_name: Network name for determining the saving location
    :param feature_folder: For saving the network with a particular feature
    :param path: Path where the network is saved to. If not set, the default is used
    :param use_cwd: If set to True, the current directory is added to the path
    :return: None
    """
    connect = nest.GetConnections(net.torus_layer_nodes)
    connect = tuple([(c[0], c[1]) for c in connect if c[1] in net.torus_layer_nodes])
    net_dict = {
        "neurons": [tuple(net.torus_layer_nodes)],
        "inh_neurons": [tuple(net.torus_inh_nodes)],
        "positions": [net.torus_layer_positions],
        "tuning_neuron": [net.tuning_to_neuron_map],
        "neuron_tuning": [net.neuron_to_tuning_map],
        "color_map": [tuple(net.color_map.reshape(-1))],
        "connect": [connect],
    }

    net_df = pd.DataFrame(net_dict)

    if path == "":
        curr_dir = os.getcwd()
        if feature_folder != "":
            path = "%s/network_files/models/%s/%s" % (curr_dir, network_name, feature_folder)
        else:
            path = "%s/network_files/models/%s" % (curr_dir, network_name)
        Path(path).mkdir(parents=True, exist_ok=True)
    elif use_cwd:
        curr_dir = os.getcwd()
        path = curr_dir + path + network_name

    num = len(os.listdir(path))
    net_df.to_csv("%s/%s_%s.csv" % (path, network_name, num), encoding='utf-8', index=False)


def load_net(net, network_name, feature_folder="", path="", use_cwd=True, num=None):
    """
        Load the neurons and connections of the network from a file
        :param net: The network object
        :param network_name: Network name for determining the location where the file is saved
        :param feature_folder: For saving the network with a particular feature
        :param path: Path where the network is loaded from. If not set, the default is used
        :param use_cwd: If set to True, the current directory is added to the path
        :param num: Determines the network file number that is to be loaded. If set to None, a network is randomly
        chosen
        :return: None
        """
    if path == "":
        curr_dir = os.getcwd()
        if feature_folder != "":
            path = "%s/network_files/models/%s/%s" % (curr_dir, network_name, feature_folder)
        else:
            path = "%s/network_files/models/%s" % (curr_dir, network_name)
    elif use_cwd:
        curr_dir = os.getcwd()
        path = curr_dir + path + network_name
        path += "/%s" % feature_folder if feature_folder != "" else ""

    if num is None:
        max_num = len(os.listdir(path))
        num = np.random.randint(0, max_num)

    net_df = pd.read_csv("%s/%s_%s.csv" % (path, network_name, num), usecols=["positions"])

    net.torus_layer_positions = [convert_string_to_list(coordinates, float)
                                 for coordinates in net_df["positions"][0].strip("(").strip(")").split("), (")]
    del net_df

    if net.verbosity > 0:
        print_msg("Create nodes")

    net.torus_layer, net.spike_detect, net.multi_meter, net.spike_gen = nc.create_torus_layer_uniform(
        num_neurons=net.num_sensory,
        size_layer=net.layer_size,
        rest_pot=net.pot_reset,
        threshold_pot=net.pot_threshold,
        time_const=net.time_constant,
        capacitance=net.capacitance,
        bg_rate=net.bg_rate,
        p_rf=net.p_rf,
        ff_factor=net.ff_factor,
        synaptic_strength=net.ff_weight,
        positions=net.torus_layer_positions,
        to_file=net.to_file
    )
    net.torus_layer_tree = KDTree(net.torus_layer_positions)

    net_df = pd.read_csv("%s/%s_%s.csv" % (path, network_name, num), usecols=["neurons", "inh_neurons"])
    net.torus_layer_nodes = convert_string_to_list(net_df["neurons"][0], dtype=int)
    net.num_sensory = len(net.torus_layer_nodes)
    net.torus_inh_nodes = convert_string_to_list(net_df["inh_neurons"][0], int)
    del net_df

    net_df = pd.read_csv("%s/%s_%s.csv" % (path, network_name, num), usecols=["neuron_tuning", "tuning_neuron"])
    net.neuron_to_tuning_map = ast.literal_eval(net_df["neuron_tuning"][0])
    net.tuning_to_neuron_map = ast.literal_eval(net_df["tuning_neuron"][0])
    del net_df

    net_df = pd.read_csv("%s/%s_%s.csv" % (path, network_name, num), usecols=["color_map"])
    net.color_map = np.asarray(convert_string_to_list(net_df["color_map"][0], dtype=int)).reshape(
        (int(net.layer_size / net.spacing_perlin), int(net.layer_size / net.spacing_perlin))
    )
    del net_df

    net.create_retina()

    if net.verbosity > 0:
        print_msg("Create connections")

    net_df = pd.read_csv("%s/%s_%s.csv" % (path, network_name, num), usecols=["connect"])
    conns = [convert_string_to_list(c, int)
             for c in net_df["connect"][0].strip("(").strip(")").split("), (")]
    del net_df

    for s, t in conns:
        weight = net.inh_weight if s in net.torus_inh_nodes else net.cap_s
        nest.Connect([s], [t], syn_spec={"weight": weight})


