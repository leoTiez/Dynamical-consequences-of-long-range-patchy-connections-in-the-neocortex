#!/usr/bin/python3

import os
import numpy as np
import cvxpy as cvx
from scipy.fft import idct, dct, fft
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import nest

nest.Install("nestmlmodule")

VERBOSITY = 3


def load_image(name, path=None):
    if path is None:
        path = os.getcwd() + "/test-input/"
    image = Image.open(path + name).convert("L")
    return np.asarray(image)


def idct2(x):
    return idct(idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def observations_from_linear_model(
        firing_rates,
        sensor_connect_mat,
        connection_strength,
        time_const=20.0,
        threshold_pot=1.,
        rest_pot=0.,
):
    return ((time_const / 1000.) * firing_rates + 0.5) * (threshold_pot - rest_pot) -\
           (connection_strength * sensor_connect_mat.dot(firing_rates))  # Divide time constant by 1000 to obtain secs?


def stimulus_reconstruction(
        firing_rates,
        sensor_connection_strength,
        receptor_connection_strength,
        receptor_sensor_connect,
        sensor_connect_mat,
        time_const=20.0,
        threshold_pot=1.,
        rest_pot=0.,
        stimulus_size=1e4
):
    cosine_tranform = dct(np.identity(int(np.sqrt(stimulus_size))), norm="ortho", axis=0)
    kron_cosine = np.kron(cosine_tranform, cosine_tranform)
    inv_kron_cosing = np.linalg.inv(kron_cosine)
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
    sampling_trans_mat = receptor_connection_strength * receptor_sensor_connect.T.dot(inv_kron_cosing)
    constraints = [sampling_trans_mat * optimisation_vector == observations]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    stimulus_pixel_matrix = np.array(optimisation_vector.value).squeeze()
    stimulus_pixel_matrix.reshape(int(np.sqrt(stimulus_size)), int(np.sqrt(stimulus_size)))
    stimulus = idct2(stimulus_pixel_matrix)
    plt.imshow(stimulus.reshape(int(np.sqrt(stimulus_size)), int(np.sqrt(stimulus_size))), cmap='gray')
    plt.show()
    return stimulus


def create_input_current_generator(
        input_stimulus,
        num_receptors=1e4
):
    assert np.all(input_stimulus < 256) and np.all(input_stimulus >= 0)

    # Transforming to k-sparse stimulus ---> Necessary?

    # fourier_trans_input = fft(trans_input)
    # trans_input[np.abs(fourier_trans_input) <= epsilon] = 0  # How to transform?

    current_dict = [{"amplitude": float(amplitude)} for amplitude in input_stimulus.reshape(-1)]

    dc_generator = nest.Create("dc_generator", n=int(num_receptors), params=current_dict)
    return dc_generator


def create_sensory_nodes(
        num_neurons=1e3,
        time_const=20.0,
        rest_pot=0.0,
        threshold_pot=1.0,
        capacitance=1.,
        use_barranca=True
):
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
    connect_dict = {
        "rule": "fixed_indegree",
        "indegree": indegree
    }

    synapse_dict = {
        "weight": connection_strength
    }

    nest.Connect(src_nodes, target_nodes, conn_spec=connect_dict, syn_spec=synapse_dict)


def main():
    image = load_image("dots50.png")
    if VERBOSITY > 2:
        plt.imshow(image, cmap='gray')
        plt.show()

    simulation_time = 250.0
    num_receptors = image.size
    num_sensors = num_receptors // 10
    indegree_rec_sen = 5
    indegree_sen_sen = 25
    cap_s = 1.
    receptor_connect_strength = 1.
    num_sensor_connections = indegree_sen_sen * num_sensors
    threshold_pot = 1.
    capacitance = 1.

    receptor_nodes = create_input_current_generator(image, num_receptors=num_receptors)
    sensory_nodes, spike_detect, multi_meter = create_sensory_nodes(
        num_neurons=num_sensors,
        threshold_pot=threshold_pot,
        capacitance=capacitance,
        use_barranca=True
    )

    create_connections_random(
        receptor_nodes,
        sensory_nodes,
        connection_strength=receptor_connect_strength,
        indegree=indegree_rec_sen,
    )
    receptor_sensor_connect_values = nest.GetConnections(receptor_nodes, sensory_nodes)
    receptor_sensor_mat = np.zeros((int(num_receptors), int(num_sensors)))
    for n in receptor_sensor_connect_values:
        receptor_sensor_mat[n[0] - min(receptor_nodes), n[1] - min(sensory_nodes)] = 1

    create_connections_random(
        sensory_nodes,
        sensory_nodes,
        connection_strength=cap_s/float(num_sensor_connections),
        indegree=indegree_sen_sen
    )
    sensor_connect_values = nest.GetConnections(sensory_nodes, sensory_nodes)
    sensor_mat = np.zeros((int(num_sensors), int(num_sensors)))
    for n in sensor_connect_values:
        sensor_mat[n[0] - min(sensory_nodes), n[1] - min(sensory_nodes)] = 1

    if VERBOSITY > 0:
        print("\n#####################\t"
              "The estimate N_A %s and the actual number of sensory neuron connections %s\n"
              % (num_sensor_connections, sensor_mat.sum()))
    nest.Simulate(simulation_time)

    data_sp = nest.GetStatus(spike_detect, keys="events")[0]
    spikes_s = data_sp["senders"]
    time_s = data_sp["times"]
    if VERBOSITY > 1:
        plt.plot(time_s, spikes_s, "g.")
        plt.show()

    spike_count = Counter(spikes_s)
    firing_rates = np.zeros(len(sensory_nodes))
    for value, number in spike_count.items():
        firing_rates[int(value) - min(sensory_nodes)] = number / float(simulation_time / 1000.)

    if VERBOSITY > 0:
        average_firing_rate = np.mean(firing_rates)
        print("\n#####################\tAverage firing rate: %s \n" % average_firing_rate)

    stimulus_reconstruction(
        firing_rates,
        cap_s/float(num_sensor_connections),
        receptor_connect_strength,
        receptor_sensor_mat,
        sensor_mat,
        stimulus_size=num_receptors,
        threshold_pot=threshold_pot
    )


if __name__ == '__main__':
    main()

