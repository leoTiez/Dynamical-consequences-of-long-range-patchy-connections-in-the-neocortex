#!/usr/bin/python3
import os

from createThesisNetwork import NETWORK_TYPE
from modules.createStimulus import INPUT_TYPE
from modules.thesisUtils import arg_parse
from ThesisReconstructionMeasure import PARAMETER_DICT


def main_experiment_loop(
        parameter=PARAMETER_DICT["tuning"],
        img_prop=[1.0],
        num_trials=10,
        spatial_sampling=False
):
    """
    Outer loop for experiments.
    :param parameter: Parameter under investigation
    :param img_prop: Sampling rate
    :param num_trials: Number of trials
    :param spatial_sampling: Flag, if set to true, the sampling of the neurons is dependent on their spatial correlation
    :return: None
    """
    curr_dir = os.getcwd()

    # #############################################################################################################
    # Looping over network and input types
    # #############################################################################################################
    for network in NETWORK_TYPE.keys():
        if network.lower() == "random" and parameter.lower() == "cluster":
            continue
        if "patchy" not in network.lower() and parameter.lower() == "patches":
            continue

        input_list = INPUT_TYPE.keys()
        if parameter.lower() == "perlin":
            input_list = ["perlin"]
        for stimulus in input_list:
            for ip in img_prop:
                os.system("python3 %s/ThesisReconstructionMeasure.py "
                          "--network=%s "
                          "--input=%s "
                          "--parameter=%s "
                          "--img_prop=%s "
                          "--num_trials=%s "
                          "%s"
                          % (
                              curr_dir,
                              network,
                              stimulus,
                              parameter,
                              ip,
                              num_trials,
                              "--spatial_sampling" if spatial_sampling else ""
                          )
                          )


def main():
    # #############################################################################################################
    # Parsing command line arguments
    # #############################################################################################################
    cmd_params = arg_parse()

    parameter = PARAMETER_DICT["tuning"]
    img_prop = None
    spatial_sampling = False

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

    # #############################################################################################################
    # Running experimental loop
    # #############################################################################################################
    main_experiment_loop(
        parameter=parameter,
        img_prop=img_prop,
        num_trials=num_trials,
        spatial_sampling=spatial_sampling
    )


if __name__ == '__main__':
    main()

