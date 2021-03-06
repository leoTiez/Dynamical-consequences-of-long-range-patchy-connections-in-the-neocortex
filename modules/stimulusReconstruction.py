#!/usr/bin/python3
import cvxpy as cvx

from modules.thesisUtils import *
import numpy as np
from scipy.fftpack import idct, fft2


def _observations_from_linear_model(
        firing_rates,
        sensor_connect_mat,
        connection_strength,
        time_const=20.0,
        threshold_pot=1e3,
        rest_pot=0.,
        tuning_weight_vector=None
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
    if tuning_weight_vector is None:
        tuning_weight_vector = np.ones(firing_rates.shape)
    return ((time_const / 1000.) * firing_rates + 0.5) * (threshold_pot - rest_pot) -\
           (connection_strength * sensor_connect_mat.dot(firing_rates * tuning_weight_vector))


def stimulus_reconstruction(
        firing_rates,
        sensor_connection_strength,
        receptor_connection_strength,
        receptor_sensor_connect_mat,
        sensor_connect_mat,
        time_const=20.0,
        threshold_pot=1e3,
        rest_pot=0.,
        stimulus_size=1e4,
        tuning_weight_vector=None,
        verbosity=True
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
    :param tuning_weight_vector: The weight vector for the neurons with stimulus preference, e.g. feature class / #class
    :param verbosity: Boolean parameter to set whether to show output from solver
    :return: Reconstructed stimulus
    """
    # Use that for the Kronecker product inv(A kron B) == inv(A) kron inv(B)
    cosine_tranform = idct(np.identity(int(np.sqrt(stimulus_size))), norm="ortho", axis=0)
    kron_cosine = np.kron(cosine_tranform, cosine_tranform)
    # Compute transformed observations based on the linearisation of the model for the L1 optimisation procedure
    observations = _observations_from_linear_model(
        firing_rates=firing_rates,
        time_const=time_const,
        threshold_pot=threshold_pot,
        rest_pot=rest_pot,
        sensor_connect_mat=sensor_connect_mat,
        connection_strength=sensor_connection_strength,
        tuning_weight_vector=tuning_weight_vector
    )

    # The L1 optimisation
    optimisation_vector = cvx.Variable(int(stimulus_size))
    objective = cvx.Minimize(cvx.norm(optimisation_vector, 1))
    sampling_trans_mat = receptor_connection_strength * receptor_sensor_connect_mat.T.dot(kron_cosine)
    constraints = [sampling_trans_mat * optimisation_vector == observations]
    prob = cvx.Problem(objective, constraints)
    _ = prob.solve(verbose=verbosity)
    stimulus_pixel_matrix = np.array(optimisation_vector.value).squeeze()
    stimulus_pixel_matrix = stimulus_pixel_matrix.reshape(int(np.sqrt(stimulus_size)), int(np.sqrt(stimulus_size))).T
    stimulus = idct2(stimulus_pixel_matrix)

    return stimulus


def direct_stimulus_reconstruction(
        firing_rates,
        ff_weight_mat
):
    """
    Reconstruction of stimulus based on the knowledge of stimulus tuning of neurons
    :param firing_rates: Firing rates of the neurons
    :param ff_weight_mat: Weight matrix of the feedforward weights
    :return: Reconstructed stimulus
    """
    weight_mat = ff_weight_mat[:, None]
    inv_ff_weight_mat = np.linalg.pinv(weight_mat)
    firing_rates_int = np.ones(firing_rates.size + 1)
    firing_rates_int[:firing_rates.size] = firing_rates
    reconstruction = firing_rates_int.dot(inv_ff_weight_mat)
    reconstruction = reconstruction.reshape(-1)[:-1]
    return reconstruction.reshape(int(np.sqrt(reconstruction.size)), int(np.sqrt(reconstruction.size)))


def oblivious_stimulus_reconstruction(
        firing_rates,
        input_neuron_mask,
        ff_adj_matrix,
        tuning_vector,
        fr_min=2.,
):
    """
    The reconstruction method if no information about the input transformation is known.
    :param firing_rates: The firing rates
    :param input_neuron_mask: Mask for neurons that receive feedforward input
    :param ff_adj_matrix: The feedforward adjacency matrix
    :param tuning_vector: The tuning vector
    :param fr_min: The minimum firing rate that is said to be influential
    :return: The reconstructed stimulus
    """
    fr = np.zeros(firing_rates[input_neuron_mask].size)
    if fr_min is None:
        fr_min = np.mean(firing_rates[input_neuron_mask])

    threshold_mask = firing_rates[input_neuron_mask] > fr_min
    fr[threshold_mask] = firing_rates[input_neuron_mask][threshold_mask]
    class_transformation = (tuning_vector + 1) / (tuning_vector.max() + 1)
    # Set no transformation for inhibitory neurons
    class_transformation[class_transformation <= 0] = 1.
    feat_trans_fr = fr * class_transformation[input_neuron_mask]
    reconstruction = ff_adj_matrix.dot(feat_trans_fr)
    reconstruction = np.nan_to_num(reconstruction / ff_adj_matrix.dot(fr))
    return reconstruction.reshape(int(np.sqrt(reconstruction.size)), int(np.sqrt(reconstruction.size)))


def fourier_trans(signal):
    """
    Fourier transformation of a two dimensional signal, e.g. image
    :param signal: The input image signal
    :return: Fourier transformation of the image
    """
    return fft2(signal)




