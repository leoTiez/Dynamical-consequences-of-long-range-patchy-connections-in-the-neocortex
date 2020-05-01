#!/usr/bin/python3
import os
import sys
import multiprocessing
from modules.thesisUtils import arg_parse
from modules.thesisConstants import  NETWORK_TYPE


def main():
    less_cpus = 2
    num_neurons = 10000
    patches = 3
    c_alpha = 0.7
    num_trials = 10

    cmd_params = arg_parse(sys.argv[1:])
    if cmd_params.num_neurons is not None:
        num_neurons = int(cmd_params.num_neurons)

    if cmd_params.patches is not None:
        patches = cmd_params.patches

    if cmd_params.c_alpha is not None:
        c_alpha = cmd_params.c_alpha

    if cmd_params.num_trials is not None:
        num_trials = cmd_params.num_trials

    if cmd_params.less_cpus is not None:
        less_cpus = cmd_params.less_cpus

    curr_dir = os.getcwd()

    for network in NETWORK_TYPE.keys():
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - less_cpus or 1)
        pool.apply_async(
            os.system,
            args=(
                "python3 %s/CreateNetworkSamples.py "
                "--network=%s "
                "--num_neurons=%s "
                "--num_trials=%s "
                "--patches=%s "
                "--c_alpha=%s "
                "--num_trials=%s "
            ) % (
                curr_dir,
                network,
                num_neurons,
                num_trials,
                patches,
                c_alpha,
                num_trials
            )
        )

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()

