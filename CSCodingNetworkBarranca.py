#!/usr/bin/python3

# ####################################################################################################################
# This Python script implements the input stimulus reconstruction based on the neural response of a network of
# sensory neurons, as described in the paper by Barranca et al.:
#
# Barranca VJ, Kovačič G, Zhou D, Cai D
# Sparsity and Compressed Coding in Sensory Systems
# PLOS Computational Biology 10(8): e1003793. (2014)
# https://doi.org/10.1371/journal.pcbi.1003793
# ####################################################################################################################
import os
import numpy as np
import cvxpy as cvx
from scipy.fft import idct
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import nest

# Import customised neural model
nest.Install("nestmlmodule")

VERBOSITY = 2


def load_image(name, path=None):
    """
    Load image with given name from path
    :param name: Name with suffix of the picture
    :param path: Path to the image. If None is passed, the current directory + '/test-input/' is taken
    :return: The image as numpy array
    """
    if path is None:
        path = os.getcwd() + "/test-input/"
    image = Image.open(path + name).convert("L")
    return np.asarray(image)


def mutual_information_hist(joint_hist):
    """
    Compute mutual information based on histogram of arrays
    :param joint_hist: The joint histogram of the input image and the reconstructed stimulus
    :return: The mutual information between input image and reconstructed stimulus
    """
    # Bins count into probability values
    joint_xy_p = joint_hist / float(np.sum(joint_hist))
    # Marginal probabilities
    marginal_x_p = np.sum(joint_xy_p, axis=1)
    marginal_y_p = np.sum(joint_xy_p, axis=0)
    # Broadcasting
    mult_xy_p = marginal_x_p[:, None] * marginal_y_p[None, :]
    non_zero_indices = joint_xy_p > 0
    return np.sum(joint_xy_p[non_zero_indices] * np.log(joint_xy_p[non_zero_indices] / mult_xy_p[non_zero_indices]))


def idct2(x):
    """
    Two dimensional inverse discrete cosine transform
    :param x: Input array
    :return: The two-dim array computed through the two-dim inverse discrete cosune transform
    """
    return idct(idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


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

    if VERBOSITY > 1:
        plt.imshow(stimulus.reshape(int(np.sqrt(stimulus_size)), int(np.sqrt(stimulus_size))), cmap='gray')
        plt.show()

    return stimulus


def create_input_current_generator(
        input_stimulus
):
    """
    Create direct current generator to simulate input stimulus. The pixel values of the image are transformed
    to an integer value representing the intensity in Ampere A
    :param input_stimulus: Grayscale input stimulus with integer values between 0 and 256
    :return: Tuple with ids for the dc generator devices
    """
    assert np.all(input_stimulus < 256) and np.all(input_stimulus >= 0)

    num_receptors = input_stimulus.size
    # Multiply value with 1e12, as the generator expects values in pA
    current_dict = [{"amplitude": float(amplitude * 1e12)} for amplitude in input_stimulus.reshape(-1)]
    dc_generator = nest.Create("dc_generator", n=int(num_receptors), params=current_dict)
    return dc_generator


def create_sensory_nodes(
        num_neurons=1e3,
        time_const=20.0,
        rest_pot=0.0,
        threshold_pot=1e3,
        capacitance=1e12,
        use_barranca=True
):
    """
    Create the sensory nodes of the network
    :param num_neurons: Number of sensory nodes that have to be created
    :param time_const: Membrane time constant in ms
    :param rest_pot: Resting potential / reset potential in mV
    :param threshold_pot: Threshold potential in mV
    :param capacitance: Capacitance of the membrane in pF
    :param use_barranca: Flag determining whether the customised Barranca neuron should be used
    :return: Tuple with ids of the neurons
    """
    if use_barranca:
        neuron_dict_barranca = {
            "tau_m": time_const,
            "V_th": threshold_pot,
            "V_R": rest_pot,
            "C_m": capacitance
        }
        sensory_nodes = nest.Create("barranca_neuron", n=int(num_neurons), params=neuron_dict_barranca)
    else:
        neuron_dict_iaf_delta = {
            "V_m": rest_pot,
            "E_L": rest_pot,
            "C_m": capacitance,
            "tau_m": time_const,
            "V_th": threshold_pot,
            "V_reset": rest_pot
        }
        sensory_nodes = nest.Create("iaf_psc_delta", n=int(num_neurons), params=neuron_dict_iaf_delta)

    # Create spike detector for stimulus reconstruction
    spikedetector = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
    multimeter = nest.Create("multimeter", params={"withtime": True, "record_from": ["V_m"]})
    nest.Connect(sensory_nodes, spikedetector)
    nest.Connect(multimeter, sensory_nodes)
    return sensory_nodes, spikedetector, multimeter


def create_connections_random(
        src_nodes,
        target_nodes,
        indegree=10,
        connection_strength=0.7,
):
    """
    Create synaptic connections from source to target nodes that are not limited in the area (i.e. the function does
    not consider receptive fields)
    :param src_nodes: Source nodes
    :param target_nodes: Target nodes
    :param indegree: Number of pre-synaptic nodes per neuron
    :param connection_strength: Synaptic weight of the connections
    """
    connect_dict = {
        "rule": "fixed_indegree",
        "indegree": indegree
    }

    synapse_dict = {
        "weight": connection_strength
    }

    nest.Connect(src_nodes, target_nodes, conn_spec=connect_dict, syn_spec=synapse_dict)


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
        # In the recent state: The Barranca neuron does not perform as well as the predefined iaf neuron with a
        # delta spike
        use_barranca = False

        # Create network nodes
        receptor_nodes = create_input_current_generator(image)
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
        receptor_sensor_connect_values = nest.GetConnections(receptor_nodes, sensory_nodes)
        receptor_sensor_mat = np.zeros((int(num_receptors), int(num_sensors)))
        for n in receptor_sensor_connect_values:
            receptor_sensor_mat[n[0] - min(receptor_nodes), n[1] - min(sensory_nodes)] = 1

        # Create sensory-to-sensory-node connections
        create_connections_random(
            sensory_nodes,
            sensory_nodes,
            connection_strength=cap_s/float(num_sensor_connections),
            indegree=indegree_sen_sen
        )
        # Create connection matrix A for synapses from sensory nodes to sensory nodes
        sensor_connect_values = nest.GetConnections(sensory_nodes, sensory_nodes)
        sensor_mat = np.zeros((int(num_sensors), int(num_sensors)))
        for n in sensor_connect_values:
            sensor_mat[n[0] - min(sensory_nodes), n[1] - min(sensory_nodes)] = 1

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
        spike_count = Counter(spikes_s)
        firing_rates = np.zeros(len(sensory_nodes))
        for value, number in spike_count.items():
            firing_rates[int(value) - min(sensory_nodes)] = number / float(simulation_time / 1000.)

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
        reconstruction_data.append(reconstruction.reshape(-1))

    # If the flag is set compute MI between the input stimulus and the reconstructed stimulus
    if compute_mi:
        hist_2d, _, _ = np.histogram2d(np.mean(input_data, axis=0), np.mean(reconstruction_data, axis=0))
        mutual_information = mutual_information_hist(hist_2d)
        print("\n#####################\tMutual Information MI: %s \n" % mutual_information)


if __name__ == '__main__':
    main(compute_mi=False)
