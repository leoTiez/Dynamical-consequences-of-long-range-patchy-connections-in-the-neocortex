#!/usr/bin/python3
import os
import numpy as np
import pandas as pd

from createThesisNetwork import NETWORK_TYPE
from modules.networkConstruction import TUNING_FUNCTION


def check_naming(file_name, file_split, network, stimulus):
    is_patchy = "patchy" in file_name
    if "patchy" in network:
        is_network = network in file_name and is_patchy
    elif "random" in network:
        is_network = network in file_split
    elif len(network) == 0:
        is_network = True
    else:
        is_network = network in file_name and not is_patchy
    # use "in" to allow empty strings
    is_stimulus = stimulus in file_split[-9]
    return is_network, is_stimulus


def check_network(file_name):
    for net in reversed(list(NETWORK_TYPE.keys())):
        if net in file_name:
            return net


def check_stimulus(file_name, network):
    idx = len(network)
    input_type = file_name[idx + 1:].split("_")[0]
    return input_type


def check_measure_type(file_name):
    if "error_distance.txt" in file_name:
        return "distance"
    elif "mean_error.txt" in file_name:
        return "mean"
    elif "error_variance.txt" in file_name:
        return "variance"


def check_sampling_rate(file_name):
    img_prop_str = "img_prop"
    idx = file_name.index(img_prop_str)
    num_letters = len(img_prop_str)
    return file_name[idx + num_letters + 1: idx + num_letters + 4]


def check_experiment_type(file_name):
    experiment_type = ""
    offset = 0
    if "orientation_map" in file_name:
        experiment_type = "orientation_map"
        offset = 1

    elif "tuning_function" in file_name:
        experiment_type = "tuning_function"
        offset = 1

    elif "num_patches" in file_name:
        experiment_type = "num_patches"
        offset = 1

    elif "perlin_cluster_size" in file_name:
        experiment_type = "perlin_cluster_size"
        offset = 1

    elif "weight_balance" in file_name:
        experiment_type = "weight_balance"
        offset = 1

    num_letters = len(experiment_type)
    idx = file_name.index(experiment_type)
    experiment_parameter = file_name[idx + num_letters + offset:].split("_")[0]
    if experiment_type == "tuning_function":
        if experiment_parameter not in TUNING_FUNCTION.keys():
            experiment_parameter = list(TUNING_FUNCTION.keys())[int(experiment_parameter)]

    return experiment_type, experiment_parameter


def table_setup(data_dict, network, stimulus, sampling_rate, experiment_type, experiment_parameter):
    temp_dict = {stimulus: {experiment_type: {sampling_rate: {experiment_parameter: {
        "distance": [],
        "variance": 0,
        "mean": 0
    }}}}}
    if network not in data_dict.keys():
        data_dict[network] = temp_dict
    elif stimulus not in data_dict[network].keys():
        data_dict[network][stimulus] = temp_dict[stimulus]
    elif experiment_type not in data_dict[network][stimulus].keys():
        data_dict[network][stimulus][experiment_type] = temp_dict[stimulus][experiment_type]
    elif sampling_rate not in data_dict[network][stimulus][experiment_type]:
        data_dict[network][stimulus][experiment_type][sampling_rate] = temp_dict[stimulus][experiment_type][
            sampling_rate]
    elif experiment_parameter not in data_dict[network][stimulus][experiment_type][sampling_rate].keys():
        data_dict[network][stimulus][experiment_type][sampling_rate][experiment_parameter] = temp_dict[stimulus][
            experiment_type][sampling_rate][experiment_parameter]

    return data_dict


def read_files(path, add_cwd=True):
    if add_cwd:
        path = os.getcwd() + "/" + path

    data_dict = []
    file_names = sorted(os.listdir(path=path))
    for fn in file_names:
        network = check_network(fn)
        stimulus = check_stimulus(fn, network)
        sampling_rate = check_sampling_rate(fn)
        measure = check_measure_type(fn)
        experiment_type, experiment_parameter = check_experiment_type(fn)

        # data_dict = table_setup(data_dict, network, stimulus, sampling_rate, experiment_type, experiment_parameter)
        file = open(path + "/" + fn, "r")
        value = float(file.read())
        file.close()
        data_dict.append([network, stimulus, experiment_type, sampling_rate, experiment_parameter, measure, value])

    df = pd.DataFrame(data_dict)
    df.rename(columns={
            0: "network",
            1: "stimulus",
            2: "experiment",
            3: "sampling",
            4: "parameter",
            5: "measure",
            6: "value"
        }, inplace=True)

    return df



