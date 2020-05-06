#!/usr/bin/python3
import numpy as np

GLOBAL_CONNECTIVITY = 0.0123
R_MAX = 8.

IMG_PROP = [0.8, 0.6, 0.4]
PERLIN_INPUT = [4, 8, 16, 50]
REC_FACTORS_PAR = [0.6, 0.8, 1.2, 1.4]
PATCHES_PAR = np.arange(1, 10, 1)
FUNC_MAP_CLUSTER_PAR = [(4, 4), (8, 8), (12, 12), (16, 16), (20, 20)]
ALPHA_PAR = np.arange(0.5, 1.0, 0.1)

NETWORK_TYPE = {
    "random": 0,
    "local_circ": 1,
    "local_sd": 2,
    "local_circ_patchy_sd": 3,
    "local_circ_patchy_random": 4,
    "local_sd_patchy_sd": 5,
    "input_only": 6
}

INPUT_TYPE = {
    "plain": 0,
    "perlin": 1,
    "bar": 2,
    "circles": 3,
    "natural": 4,
    "edges": 5,
    "random": 6
}

PARAMETER_DICT = {
    "tuning": 0,
    "cluster": 1,
    "patches": 2,
    "weights": 3,
    "alpha": 4
}

TUNING_FUNCTION = {
    "step": 0,
    "gauss": 1,
    "linear": 2
}

