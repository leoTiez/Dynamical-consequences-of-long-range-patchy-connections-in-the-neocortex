#!/usr/bin/python3
from modules.thesisUtils import *

import numpy as np
from scipy.fft import idct
import matplotlib.pyplot as plt
import cvxpy as cvx


def observations_from_linear_model(
        firing_rates,
        sensor_connect_mat,
        connection_strength,
        time_const=20.0,
        threshold_pot=1e3,
        rest_pot=0.,
):
    """
    Compute the observations from the linearised neural activity for stimulus reconstruction.
    The computation is based on the paper by Barranca et al. (see above).
    :param firing_rates: The firing rates of the neurons obtained from the network response
    :param sensor_connect_mat: Connection matrix A of the sensory to sensory neurons. A_ij = 1 if there exists a
     connection between node i and j and 0 otherwise.
    :param connection_strength: Weight of the connections between sensory to sensory neurons. Note that this weight
    is equal to S / N_A where S is a multiplier for the connection strength and N_A is the number of connections
    of in the sensory connection matrix
    :param time_const: Time constant in ms
    :param threshold_pot: Threshold potential in mV
    :param rest_pot: resting potential / reset potential in mV
    :return: The observation that is used for the stimulus reconstruction
    """
    return ((time_const / 1000.) * firing_rates + 0.5) * (threshold_pot - rest_pot) -\
           (connection_strength * sensor_connect_mat.dot(firing_rates))


def stimulus_reconstruction(
        firing_rates,
        sensor_connection_strength,
        receptor_connection_strength,
        receptor_sensor_connect_mat,
        sensor_connect_mat,
        time_const=20.0,
        threshold_pot=1e3,
        rest_pot=0.,
        stimulus_size=1e4
):
    """
    Reconstruct input stimulus based on the firing patterns of the nodes in the network
    :param firing_rates: Firing rates of the individual nodes in Hz
    :param sensor_connection_strength: Synaptic weight of the connections of sensory-to-sensory-node links. Note that
    this weight is equal to S / N_A where S is a multiplier for the connection strength and N_A is the number of
    connections
    :param receptor_connection_strength: Synaptic weight of the connections of receptor-to-sensory-node links
    :param receptor_sensor_connect_mat: Connection matrix A for receptor-to-sensory-node links. A_ij = 1 if there exists
     a connection between node i and j and 0 otherwise.
    :param sensor_connect_mat: Connection matrix B for sensory-to-sensory-node links. B_ij = 1 if there exists
     a connection between node i and j and 0 otherwise.
    :param time_const: Membran time constant in ms
    :param threshold_pot: Threshold potential in mV
    :param rest_pot: resting potential in mV
    :param stimulus_size: number of the input stimulus values
    :return:
    """
    # Use that for the Kronecker product inv(A kron B) == inv(A) kron inv(B)
    cosine_tranform = idct(np.identity(int(np.sqrt(stimulus_size))), norm="ortho", axis=0)
    kron_cosine = np.kron(cosine_tranform, cosine_tranform)
    # Compute transformed observations based on the linearisation of the model for the L1 optimisation procedure
    observations = observations_from_linear_model(
        firing_rates=firing_rates,
        time_const=time_const,
        threshold_pot=threshold_pot,
        rest_pot=rest_pot,
        sensor_connect_mat=sensor_connect_mat,
        connection_strength=sensor_connection_strength
    )

    # The L1 optimisation
    optimisation_vector = cvx.Variable(int(stimulus_size))
    objective = cvx.Minimize(cvx.norm(optimisation_vector, 1))
    sampling_trans_mat = receptor_connection_strength * receptor_sensor_connect_mat.T.dot(kron_cosine)
    constraints = [sampling_trans_mat * optimisation_vector == observations]
    prob = cvx.Problem(objective, constraints)
    _ = prob.solve(verbose=True)
    stimulus_pixel_matrix = np.array(optimisation_vector.value).squeeze()
    stimulus_pixel_matrix = stimulus_pixel_matrix.reshape(int(np.sqrt(stimulus_size)), int(np.sqrt(stimulus_size))).T
    stimulus = idct2(stimulus_pixel_matrix)

    return stimulus