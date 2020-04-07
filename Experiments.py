#!/usr/bin/python3
import os

from createThesisNetwork import NETWORK_TYPE
from modules.createStimulus import INPUT_TYPE
from modules.thesisUtils import arg_parse
from ThesisReconstructionMeasure import PARAMETER_DICT


def main(params):
    curr_dir = os.getcwd()
    print("\n#####################\t Start with experiments for parameter %s" % params.parameter)

    num_trials = params.num_trials if params.num_trials is not None else 10
    img_prop = params.img_prop if params.img_prop is not None else 1.0

    for network in NETWORK_TYPE.keys():
        input_list = INPUT_TYPE.keys()
        if params.parameter.lower() == "perlin":
            input_list = ["perlin"]
        for stimulus in input_list:
            if network.lower() == "random" and params.parameter.lower() == "cluster":
                continue
            if "patchy" not in network.lower() and params.parameter.lower() == "patches":
                continue
            os.system("python3 %s/ThesisReconstructionMeasure.py "
                      "--network=%s "
                      "--input=%s "
                      "--parameter=%s "
                      "--img_prop=%s "
                      "--num_trials=%s "
                      % (curr_dir, network, stimulus, params.parameter, img_prop, num_trials)
                      )

if __name__ == '__main__':
    cmd_params = arg_parse()

    if cmd_params.parameter not in list(PARAMETER_DICT.keys()):
        raise ValueError("Please chose a valid parameter for your experiments that is under investigation")

    main(cmd_params)

