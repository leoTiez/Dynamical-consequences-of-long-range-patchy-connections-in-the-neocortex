#!/usr/bin/pyhton3
import numpy as np
import matplotlib.pyplot as plt

from createThesisNetwork import network_factory, NETWORK_TYPE
from modules.createStimulus import stimulus_factory, INPUT_TYPE
from modules.networkConstruction import TUNING_FUNCTION

import nest

MEAN_IN_OUT_DEG = 45.


def main():
    input_stimulus = stimulus_factory(INPUT_TYPE["perlin"])

    c_loc = 0.7
    c_lr = 0.3
    num_patches = 3
    num_neurons = 10000
    simulation_time = 1000.
    weight_factor = 1.8

    cap_s = 1.
    inh_weight = -5.
    ff_weight = 1.
    p_rf = .7
    pot_threshold = -55.
    pot_reset = -70.
    capacitance = 80.
    time_constant = 20.
    img_prop = 1.
    use_dc = False
    spatial_sampling = False

    network = network_factory(
        input_stimulus,
        network_type=NETWORK_TYPE["local_circ_patchy_sd"],
        num_sensory=num_neurons,
        ff_weight=ff_weight,
        ff_factor=weight_factor,
        cap_s=0.,
        inh_weight=0.,
        c_loc=c_loc,
        c_lr=c_lr,
        p_rf=p_rf,
        num_patches=num_patches,
        pot_reset=pot_reset,
        pot_threshold=pot_threshold,
        capacitance=capacitance,
        time_constant=time_constant,
        tuning_function=TUNING_FUNCTION["gauss"],
        img_prop=img_prop,
        spatial_sampling=spatial_sampling,
        use_dc=use_dc,
        save_plots=False,
        verbosity=1,
        to_file=False
    )

    network.create_network()
    firing_rate_ff, _ = network.simulate(simulation_time)

    inh_mask = np.zeros(len(network.torus_layer_nodes)).astype('bool')
    inh_mask[np.asarray(network.torus_inh_nodes) - min(network.torus_layer_nodes)] = True
    nest.SetStatus(
        nest.GetConnections(source=np.asarray(network.torus_layer_nodes)[~inh_mask].tolist()),
        {"weight": cap_s}
    )
    nest.SetStatus(
        nest.GetConnections(source=network.torus_inh_nodes),
        {"weight": inh_weight}
    )

    firing_rate_rec, _ = network.simulate(simulation_time)
    t = nest.GetStatus(network.multi_meter, "events")[0]["times"]
    neuron = np.argmax(nest.GetStatus(network.multi_meter, "events")[0]["V_m"][t < simulation_time]) % network.num_sensory
    v = nest.GetStatus(network.multi_meter, "events")[0]["V_m"][neuron::network.num_sensory]
    t = t[neuron::network.num_sensory]

    print("Average firing rate ff only: %s" % firing_rate_ff.mean())
    print("Average firing rate recurrent: %s" % firing_rate_rec.mean())
    plt.figure(figsize=(15, 8))
    plt.plot(t[t < simulation_time], v[t < simulation_time], label="FF only")
    plt.plot(t[t > simulation_time] - simulation_time, v[t > simulation_time], label="Recurrent")
    plt.title("Membrane potential of a single neuron. FF weight factor %s" % weight_factor)
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.text(0., -72, 'Average firing rate ff only: %s' % firing_rate_ff.mean(), style='italic',
            bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})
    plt.text(0., -75, 'Average firing rate with recurrent: %s' % firing_rate_rec.mean(), style='italic',
             bbox={'facecolor': 'orange', 'alpha': 0.5, 'pad': 10})
    plt.ylim(-77., -50.)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

