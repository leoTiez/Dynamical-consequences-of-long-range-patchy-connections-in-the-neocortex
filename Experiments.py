#!/usr/bin/python3
import sys
import os
import multiprocessing
from itertools import product

from createThesisNetwork import NETWORK_TYPE
from modules.createStimulus import INPUT_TYPE
from modules.thesisUtils import arg_parse
from ThesisReconstructionMeasure import PARAMETER_DICT


def main_experiment_loop(
        parameter=PARAMETER_DICT["tuning"],
        img_prop=[1.0],
        num_trials=10,
        less_cpus=2,
        spatial_sampling=False,
        load_network=False,
        existing_ok=False
):
    """
    Outer loop for experiments.
    :param parameter: Parameter under investigation
    :param img_prop: Sampling rate
    :param num_trials: Number of trials
    :param spatial_sampling: Flag, if set to true, the sampling of the neurons is dependent on their spatial correlation
    :param load_network: If set to true, the network is loaded from file
    :return: None
    """
    curr_dir = os.getcwd()

    # #############################################################################################################
    # Looping over network and input types
    # #############################################################################################################
    network_list = list(NETWORK_TYPE.keys())
    if parameter.lower() == "cluster":
        network_list = [net for net in network_list if net != "random"]
    elif parameter.lower() == "patches":
        network_list = [net for net in network_list if "patchy" in net]

    input_list = list(INPUT_TYPE.keys()) if parameter.lower() != "perlin" else ["perlin"]

    parameter_combination = product(network_list, input_list, img_prop)

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - less_cpus or 1)
    for pc in parameter_combination:
        pool.apply_async(
            os.system,
            args=("python3 %s/ThesisReconstructionMeasure.py "
                  "%s"
                  "--network=%s "
                  "--input=%s "
                  "--parameter=%s "
                  "--img_prop=%s "
                  "--num_trials=%s "
                  "%s"
                  "%s"
                  % (
                      curr_dir,
                      "--load_network " if load_network else "",
                      pc[0],
                      pc[1],
                      parameter,
                      pc[2],
                      num_trials,
                      "--existing_ok " if existing_ok else "",
                      "--spatial_sampling" if spatial_sampling else ""
                  ),
                  )
        )

    pool.close()
    pool.join()


def main():
    # #############################################################################################################
    # Parsing command line arguments
    # #############################################################################################################
    cmd_params = arg_parse(sys.argv[1:])

    parameter = PARAMETER_DICT["tuning"]
    img_prop = None
    spatial_sampling = False
    load_network = False
    less_cpus = 2
    existing_ok = False

    if cmd_params.parameter in list(PARAMETER_DICT.keys()):
        parameter = cmd_params.parameter
    else:
        raise ValueError("Please chose a valid parameter for your experiments that is under investigation")

    print("\n#####################\t Start with experiments for parameter %s" % parameter)

    num_trials = cmd_params.num_trials if cmd_params.num_trials is not None else 10
    if cmd_params.img_prop is not None:
        if str(cmd_params.img_prop) != "all":
            img_prop = [float(cmd_params.img_prop)]
        else:
            img_prop = [1.0, 0.8, 0.6, 0.4]
    else:
        img_prop = [1.0]

    if cmd_params.spatial_sampling:
        spatial_sampling = True

    if cmd_params.load_network:
        load_network = True

    if cmd_params.less_cpus is not None:
        less_cpus = cmd_params.less_cpus

    if cmd_params.existing_ok:
        existing_ok = True

    # #############################################################################################################
    # Running experimental loop
    # #############################################################################################################
    main_experiment_loop(
        parameter=parameter,
        img_prop=img_prop,
        num_trials=num_trials,
        spatial_sampling=spatial_sampling,
        less_cpus=less_cpus,
        load_network=load_network,
        existing_ok=existing_ok
    )


if __name__ == '__main__':
    main()

