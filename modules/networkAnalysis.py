#!/usr/bin/python3
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import nest


def spatial_variance(node_tree, node_pos, firing_rates, r_neighborhood=0.3):
    """
    Compute spatial variance via retrieving all nodes within a giving distance and compute the variance
    in their firing rate
    :param node_tree: The nodes positions organised in a tree
    :param node_pos: The positions of the nodes
    :param firing_rates: The firing rates
    :param r_neighborhood: The radius of the neighborhood
    :return: The variance of the firing rate withing this neighborhood
    """
    variance = []
    for pos in node_pos:
        neighborhood = np.asarray(node_tree.query_ball_point(pos, r_neighborhood))
        nbh_fr = firing_rates[neighborhood.astype('int')]
        variance.append(np.cov(nbh_fr[nbh_fr > 0]))

    return np.nanmean(np.asarray(variance))


def mutual_information_hist(input_data, firing_rates):
    """
    Compute the mutual information between the input data and the firing rates based on the established histograms
    :param input_data: The input images as a list of image vectors
    :param firing_rates: The firing rates as a list of vectors
    :return: The mutual information
    """
    hist_input, bins_input = np.histogram(np.mean(input_data, axis=0))
    hist_f_rate, bins_f_rate = np.histogram(np.mean(firing_rates, axis=0))

    prob_in = hist_input / float(np.sum(hist_input))
    prob_f_rate = hist_f_rate / float(np.sum(hist_f_rate))

    ind_prob = prob_in[:, None] * prob_f_rate[None, :]
    joint_hist, _, _ = np.histogram2d(np.mean(input_data, axis=0), np.mean(input_data, axis=0), bins=[bins_input, bins_f_rate])
    joint_prob = joint_hist / float(np.sum(joint_hist))
    non_zero_indices = joint_prob > 0
    return np.sum(joint_prob[non_zero_indices] * np.log(joint_prob[non_zero_indices] / ind_prob[non_zero_indices]))


def mutual_information_img(input_data, reconstruction_data):
    """
    Compute mutual information based on histogram of arrays
    :param input_data: The original stimulus
    :param reconstruction_data: The reconstructed stimulus
    :return: The mutual information between input image and reconstructed stimulus
    """
    joint_hist, _, _ = np.histogram2d(np.mean(input_data, axis=0), np.mean(reconstruction_data, axis=0))
    # Bins count into probability values
    joint_xy_p = joint_hist / float(np.sum(joint_hist))
    # Marginal probabilities
    marginal_x_p = np.sum(joint_xy_p, axis=1)
    marginal_y_p = np.sum(joint_xy_p, axis=0)
    # Broadcasting
    mult_xy_p = marginal_x_p[:, None] * marginal_y_p[None, :]
    non_zero_indices = joint_xy_p > 0
    return np.sum(joint_xy_p[non_zero_indices] * np.log(joint_xy_p[non_zero_indices] / mult_xy_p[non_zero_indices]))


def error_distance(input_data, reconstructed_data):
    """
    Computes the normalised Eucledian distance between the original image and the reconstructed image
    :param input_data: The input image
    :param reconstructed_data: The reconstructed image
    :return: The normalised error
    """
    # Normalise stimuli
    input_data = input_data.astype('float') / float(input_data.max())
    reconstructed_data = reconstructed_data.astype('float') / float(reconstructed_data.max())

    error = np.linalg.norm(input_data - reconstructed_data)
    normalised_error = error / np.linalg.norm(input_data)
    return normalised_error


def set_values_in_adjacency_matrix(connect_values, adj_mat, min_src, min_target, use_weights=True):
    """
    Set the values in the adjacency / weight matrix
    :param connect_values: The connections for which the value needs to be set
    :param adj_mat: The adjacency / weight matrix
    :param min_src: The minimum source id
    :param min_target: The minimum target id
    :param use_weights: Flag to determine whether weights or adjacency values are to be set
    :return: The updated adjacency / weight matrix
    """
    weights = nest.GetStatus(connect_values, "weight")
    for n, w in zip(connect_values, weights):
        if use_weights:
            adj_mat[n[0] - min_src, n[1] - min_target] = w
        elif w > 0:
            adj_mat[n[0] - min_src, n[1] - min_target] = 1

    return adj_mat


def create_adjacency_matrix(src_nodes, target_nodes):
    """
    Creates the adjacency matrix A for the connections between source and target nodes. A_ij = 1 if there is a
    connection between node i and j and 0 otherwise
    :param src_nodes: Source nodes
    :param target_nodes: Target nodes
    :return: Adjacency matrix
    """
    connect_values = nest.GetConnections(source=src_nodes)
    connect_values = [
        connection for connection in connect_values if nest.GetStatus([connection], "target")[0] in target_nodes
    ]

    adjacency_mat = np.zeros((len(src_nodes), len(target_nodes)))
    adjacency_mat = set_values_in_adjacency_matrix(connect_values, adjacency_mat, min(src_nodes), min(target_nodes))
    return adjacency_mat


def eigenvalue_analysis(matrix, plot=True, save_plot=False, fig_name=None, fig_path=None):
    """
    Compute and plot, if needed, eigenvalues and eigenvectors
    :param matrix: Matrix of which the eigenvalues/vectors should be computed
    :param plot: Flag to determine whether to create a plot of the eigenvalue spectrum
    :param save_plot: Flag to determine whether to save the plot of the eigenvalue spectrum
    :param fig_name: Name of the saved figure ending with .png
    :param fig_path: Path of the directory where the figure is saved ending with /
    :return: Eigenvalues and eigenvectors
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    if plot:
        plt.plot(eigenvalues.real, eigenvalues.imag, 'k,')
        plt.xlabel("Re($\lambda$)")
        plt.ylabel("Im($\lambda$)")
        if not save_plot:
            plt.show()
        else:
            if fig_path is None:
                curr_dir = os.getcwd()
                fig_path = curr_dir + "/figures/"
            if fig_name is None:
                fig_name = "eigenvalue_spec.png"
            plt.savefig(fig_path + fig_name)
            plt.close()

    return eigenvalues, eigenvectors


def get_firing_rates(spike_train, nodes, simulation_time):
    """
    Compute the firing rates of the nodes
    :param spike_train: Emitted spike train of the nodes
    :param nodes: Tuple or list with ids of the nodes that emitted the spike train
    :param simulation_time: Simulation time
    :return: Firing rate per node in the same order as the ids in the "nodes" parameter
    """
    spike_count = Counter(spike_train)
    firing_rates = np.zeros(len(nodes))
    for value, number in spike_count.items():
        firing_rates[int(value) - min(nodes)] = number / float(simulation_time / 1000.)

    return firing_rates


