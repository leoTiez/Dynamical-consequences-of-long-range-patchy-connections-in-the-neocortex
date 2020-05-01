#!/usr/bin/python3
import sys
import numpy as np

from createThesisNetwork import network_factory
from modules.createStimulus import stimulus_factory
from modules.thesisUtils import arg_parse
from modules.thesisConstants import *


def main():
    """
    Main file for creating sample networks
    :return: None
    """
    # ################################################################################################################
    # Initialise parameters that can be changed from cmd line
    # ################################################################################################################
    network_type = None
    num_neurons = int(1e4)
    tuning_function = TUNING_FUNCTION["gauss"]
    cluster = (15, 15)
    patches = 3
    c_alpha = 0.7
    ff_factor = 1.
    img_prop = 1.
    save_plots = True
    verbosity = 1
    num_trials = 10

    # ################################################################################################################
    # Parse command line arguments
    # ################################################################################################################
    cmd_params = arg_parse(sys.argv[1:])
    if cmd_params.seed:
        np.random.seed(0)

    if cmd_params.agg:
        import matplotlib
        matplotlib.use("Agg")

    if cmd_params.network == "input_only":
        raise ValueError("Cannot create network graph of network that doesn't establish recurrent connections")
    elif cmd_params.network in list(NETWORK_TYPE.keys()):
        network_type = NETWORK_TYPE[cmd_params.network]
    else:
        raise ValueError("Please pass a valid network as parameter")

    if cmd_params.num_neurons is not None:
        num_neurons = int(cmd_params.num_neurons)

    if cmd_params.patches is not None:
        patches = cmd_params.patches

    if cmd_params.c_alpha is not None:
        c_alpha = cmd_params.c_alpha

    if cmd_params.num_trials is not None:
        num_trials = cmd_params.num_trials

    # ################################################################################################################
    # Set network parameter
    # ################################################################################################################
    input_stimulus = stimulus_factory(INPUT_TYPE["plain"])
    cap_s = 1.
    inh_weight = -5.
    ff_weight = 1.0
    all_same_input_current = False
    p_rf = 0.7
    pot_threshold = -55.
    pot_reset = -70.
    capacitance = 80.
    time_constant = 20.
    use_dc = False

    for _ in range(num_trials):
        network = network_factory(
            input_stimulus,
            network_type=network_type,
            num_sensory=num_neurons,
            all_same_input_current=all_same_input_current,
            ff_weight=ff_weight,
            cap_s=cap_s,
            inh_weight=inh_weight,
            c_alpha=c_alpha,
            p_rf=p_rf,
            ff_factor=ff_factor,
            pot_reset=pot_reset,
            pot_threshold=pot_threshold,
            capacitance=capacitance,
            time_constant=time_constant,
            tuning_function=tuning_function,
            resolution_perlin=cluster,
            num_patches=patches,
            use_input_neurons=True if network_type == NETWORK_TYPE["input_only"] else False,
            img_prop=img_prop,
            spatial_sampling=False,
            use_dc=use_dc,
            save_prefix="",
            save_plots=False,
            verbosity=verbosity,
            to_file=False
        )
        network.create_network()
        network.export_net()


if __name__ == '__main__':
    main()

