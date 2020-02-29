#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ####################################################################################################################
# This Python script implements the input stimulus reconstruction based on the neural response of a network of
# sensory neurons, as described in the paper by Barranca et al.:
#
# Barranca VJ, Kovačič G, Zhou D, Cai D
# Sparsity and Compressed Coding in Sensory Systems
# PLOS Computational Biology 10(8): e1003793. (2014)
# https://doi.org/10.1371/journal.pcbi.1003793
# ####################################################################################################################

# Own libraries
from modules.stimulusReconstruction import stimulus_reconstruction
from modules.createStimulus import *
from modules.networkConstruction import *
from modules.networkAnalysis import *
from modules.thesisUtils import *

# External libraries
import numpy as np
import matplotlib.pyplot as plt

# Nest
import nest

# Import customised neural model
# nest.Install("nestmlmodule")

VERBOSITY = 2


def main(compute_mi=False):
    """
    Main function
    :param compute_mi: Flag to determine whether to compute the mutual information MI
    """
    images = ["dots50.png", "monkey50.png", "kangaroo50.png"]
    input_data = []
    reconstruction_data = []

    # If MI not interesting not necessary to iterate over all images
    if not compute_mi:
        images = ["dots50.png"]

    for image_name in images:
        image = load_image(image_name)
        if VERBOSITY > 2:
            plt.imshow(image, cmap='gray')
            plt.show()
        input_data.append(image.reshape(-1))

        # Set network parameters
        simulation_time = 250.0
        num_receptors = image.size
        num_sensors = num_receptors // 10
        indegree_rec_sen = 5
        indegree_sen_sen = 25
        cap_s = 1.
        receptor_connect_strength = 1.
        num_sensor_connections = indegree_sen_sen * num_sensors
        threshold_pot = 1e3
        capacitance = 1e12
        multiplier = 1e12
        # In the recent state: The Barranca neuron does not perform as well as the predefined iaf neuron with a
        # delta spike
        use_barranca = False

        # Create network nodes
        receptor_nodes = create_input_current_generator(image, multiplier=multiplier)
        sensory_nodes, spike_detect, multi_meter = create_sensory_nodes(
            num_neurons=num_sensors,
            threshold_pot=threshold_pot,
            capacitance=capacitance,
            use_barranca=use_barranca
        )

        # Create receptor-to-sensory-node connections
        create_connections_random(
            receptor_nodes,
            sensory_nodes,
            connection_strength=receptor_connect_strength,
            indegree=indegree_rec_sen,
        )
        # Create connection matrix B for synapses from receptors to sensory nodes
        receptor_sensor_mat = create_adjacency_matrix(receptor_nodes, sensory_nodes)

        # Create sensory-to-sensory-node connections
        create_connections_random(
            sensory_nodes,
            sensory_nodes,
            connection_strength=cap_s/float(num_sensor_connections),
            indegree=indegree_sen_sen
        )
        # Create connection matrix A for synapses from sensory nodes to sensory nodes
        sensor_mat = create_adjacency_matrix(sensory_nodes, sensory_nodes)

        if VERBOSITY > 0:
            print("\n#####################\t"
                  "The estimate N_A %s and the actual number of sensory neuron connections %s\n"
                  % (num_sensor_connections, sensor_mat.sum()))

        # Simulate network
        nest.Simulate(simulation_time)

        # Get network response in spikes
        data_sp = nest.GetStatus(spike_detect, keys="events")[0]
        spikes_s = data_sp["senders"]
        time_s = data_sp["times"]
        if VERBOSITY > 1:
            plt.plot(time_s, spikes_s, "g.")
            plt.show()

        # Count number of spikes per neuron and create an array holding the firing rates per neuron in Hz

        firing_rates = get_firing_rates(spikes_s, sensory_nodes, simulation_time)

        if VERBOSITY > 0:
            average_firing_rate = np.mean(firing_rates)
            print("\n#####################\tAverage firing rate: %s \n" % average_firing_rate)

        # Reconstruct input stimulus
        reconstruction = stimulus_reconstruction(
            firing_rates,
            cap_s/float(num_sensor_connections),
            receptor_connect_strength,
            receptor_sensor_mat,
            sensor_mat,
            stimulus_size=num_receptors,
            threshold_pot=threshold_pot
        )
        if VERBOSITY > 1:
            _, figs = plt.subplots(1, 2)
            figs[0].imshow(reconstruction, cmap="gray")
            figs[1].imshow(image, cmap="gray")
            plt.show()

        reconstruction_data.append(reconstruction.reshape(-1))

    # If the flag is set compute MI between the input stimulus and the reconstructed stimulus
    if compute_mi:
        mutual_information = mutual_information_hist(input_data, reconstruction_data)
        print("\n#####################\tMutual Information MI: %s \n" % mutual_information)


if __name__ == '__main__':
    main(compute_mi=False)
