#!/usr/bin/python3
import os
import ast
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

import modules.networkConstruction as nc

import nest


def convert_string_to_list(tuple_string, dtype=int):
    return [dtype(s) for s in tuple_string.strip("(").strip(")").split(", ")]


def save_net(net, network_name, feature_folder, path="", use_cwd=True):
    net_dict = {
        "neurons": [tuple(net.torus_layer_nodes)],
        "inh_neurons": [tuple(net.torus_inh_nodes)],
        "positions": [net.torus_layer_positions],
        "tuning_neuron": [net.tuning_to_neuron_map],
        "neuron_tuning": [net.neuron_to_tuning_map],
        "color_map": [tuple(net.color_map.reshape(-1))],
        "adj_mat": [tuple(net.adj_sens_sens_mat.reshape(-1))],
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

    net_df = pd.read_csv("%s/%s_%s.csv" % (path, network_name, num))

    net.torus_layer_nodes = convert_string_to_list(net_df["neurons"][0], dtype=int)
    net.num_sensory = len(net.torus_layer_nodes)

    net.torus_layer_positions = [convert_string_to_list(coordinates, float)
                                 for coordinates in net_df["positions"][0].strip("(").strip(")").split("), (")]

    net.adj_sens_sens_mat = np.asarray(convert_string_to_list(net_df["adj_mat"][0], float)).reshape(
        (net.num_sensory, net.num_sensory)
    )

    net.torus_layer, net.spike_detect, net.multi_meter = nc.create_torus_layer_uniform(
        num_neurons=net.num_sensory,
        size_layer=net.layer_size,
        p_rf=net.p_rf,
        ff_factor=net.ff_factor,
        synaptic_strength=net.ff_weight,
        positions=net.torus_layer_positions,
        to_file=net.to_file
    )
    net.torus_layer_tree = KDTree(net.torus_layer_positions)

    min_id = min(net.torus_layer_nodes)
    net.torus_inh_nodes = set()
    for s, row in enumerate(net.adj_sens_sens_mat):
        for t, weight in enumerate(row):
            if weight != 0:
                if weight < 0:
                    weight = net.inh_weight
                else:
                    weight = net.cap_s
                nest.Connect([s + min_id], [t + min_id], syn_spec={"weight": weight})

    net.torus_inh_nodes = convert_string_to_list(net_df["inh_neurons"][0], int)