#!/usr/bin/python3
import os
import numpy as np
import pandas as pd

from modules.thesisConstants import TUNING_FUNCTION

NETWORK_TYPE_NAMING = {
    "random": "Random",
    "local_circ": "Loc Circular",
    "local_sd": "Loc TD",
    "local_circ_patchy_sd": "Loc Circular\nTD Patches",
    "local_circ_patchy_random": "Loc Circular\nRandom Patches",
    "local_sd_patchy_sd": "Loc TD\nTD Patches",
    "input_only": "Only Input"
}


def check_network(file_name):
    """
    Returns the network type that was used for the experiment
    :param file_name: Name of the file
    :return: Network type
    """
    for net in reversed(list(NETWORK_TYPE_NAMING.keys())):
        if net in file_name:
            return net, NETWORK_TYPE_NAMING[net]


def check_stimulus(file_name, network):
    """
    Returns the stimulus type that was used for the experiment
    :param file_name: Name of the file
    :param network: Network type
    :return: Stimulus type
    """
    idx = len(network)
    input_type = file_name[idx + 1:].split("_")[0]
    return int(input_type)


def check_measure_type(file_name):
    """
    Returns what kind of measurement was written to the file
    :param file_name: Name of the file
    :return: Measurement type
    """
    if "error_distance.txt" in file_name:
        return "distance"
    elif "mean_error.txt" in file_name:
        return "mean"
    elif "error_variance.txt" in file_name:
        return "variance"


def check_sampling_rate(file_name):
    """
    Return sampling rate that was used for the experiment
    :param file_name: Name of the file
    :return: Sampling rate
    """
    img_prop_str = "img_prop"
    idx = file_name.index(img_prop_str)
    num_letters = len(img_prop_str)
    return file_name[idx + num_letters + 1: idx + num_letters + 4]


def check_experiment_type(file_name):
    """
    Return the experiment type
    :param file_name: Name of the file
    :return: Experiment type
    """
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

    elif "equilibrium" in file_name:
        experiment_type = "equilibrium"
        offset = 1

    elif "alpha" in file_name:
        experiment_type = "alpha"
        offset = 1

    num_letters = len(experiment_type)
    idx = file_name.index(experiment_type)
    experiment_parameter = file_name[idx + num_letters + offset:].split("_")[0]
    if experiment_type == "tuning_function":
        if experiment_parameter not in TUNING_FUNCTION.keys():
            experiment_parameter = list(TUNING_FUNCTION.keys())[int(experiment_parameter)]

    if experiment_type == "alpha":
        experiment_parameter = str(np.around(float(experiment_parameter), decimals=1))

    if experiment_type == "orientation_map":
        experiment_parameter = str(experiment_parameter.strip("(").strip(")").split(", ")[0])

    return experiment_type, experiment_parameter


def check_spatial_sampling(file_name):
    """
    Check whether neurons were sampled with spatial correlation
    :param file_name: Name of the file
    :return: True if it was, False else. Value returned as string
    """
    img_prop_str = "spatials"
    idx = file_name.index(img_prop_str)
    use_spatial_sampling = file_name[idx + 1:].split("_")[1]
    return use_spatial_sampling


def read_files(path, add_cwd=True):
    """
    Method that reads out the values in the files and saves them in a pandas Dataframe
    :param path: Path to the files
    :param add_cwd: If set to true, the passed path is not absolute and needs the current directory
    :return: Dataframe with all experimental data
    """
    if add_cwd:
        path = os.getcwd() + "/" + path

    data_dict = []
    file_names = sorted(os.listdir(path=path))
    for fn in file_names:
        network, network_name = check_network(fn)
        stimulus = check_stimulus(fn, network)
        sampling_rate = check_sampling_rate(fn)
        measure = check_measure_type(fn)
        experiment_type, experiment_parameter = check_experiment_type(fn)
        use_spatial_sampling = check_spatial_sampling(fn)

        file = open(path + "/" + fn, "r")
        value = float(file.read())
        file.close()
        data_dict.append([
            network_name,
            stimulus,
            experiment_type,
            sampling_rate,
            use_spatial_sampling,
            experiment_parameter,
            measure,
            value
        ])

    df = pd.DataFrame(data_dict)
    df.rename(columns={
            0: "network",
            1: "stimulus",
            2: "experiment",
            3: "sampling",
            4: "spatial_sampling",
            5: "parameter",
            6: "measure",
            7: "value"
        }, inplace=True)

    return df



