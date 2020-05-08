#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from webcolors import hex_to_rgb


from modules.createStimulus import stimulus_factory
from createThesisNetwork import network_factory
from modules.thesisConstants import *
from modules.thesisUtils import *
from modules.stimulusReconstruction import oblivious_stimulus_reconstruction

import nest


def main():
    # #################################################################################################################
    # Define values
    # #################################################################################################################
    use_single_neuron = True
    network_type_id = "local_circ_patchy_sd"
    network_type = NETWORK_TYPE[network_type_id]
    network_name = "Local Circular Network with Tuning Dependent Patches"
    num_neurons = 3000
    img_prop = 0 if use_single_neuron else 0.4
    bg_rate = 500.
    max_firing_rate = 1e4

    save_plots = False
    save_prefix = network_type_id

    cap_s = 1.
    inh_weight = -5.
    ff_weight = 1.0
    all_same_input_current = False
    p_rf = 0.7
    c_alpha = 0.5
    ff_factor = 1.
    pot_threshold = -55.
    pot_reset = -70.
    capacitance = 80.
    time_constant = 20.
    presentation_time = 0.
    resolution_func_map = (20, 20)
    num_patches = 3
    spatial_sampling = False
    use_dc = False

    min_mem_pot = 10.

    # #################################################################################################################
    # Create network
    # #################################################################################################################
    # Note: when using the same input current for all neurons, we obtain synchrony, and due to the refactory phase
    # all recurrent connections do not have any effect
    network = network_factory(
        network_type=network_type,
        num_sensory=num_neurons,
        all_same_input_current=all_same_input_current,
        ff_weight=ff_weight,
        cap_s=cap_s,
        inh_weight=inh_weight,
        c_alpha=c_alpha,
        p_rf=p_rf,
        ff_factor=ff_factor,
        max_spiking=max_firing_rate,
        bg_rate=bg_rate,
        pot_reset=pot_reset,
        pot_threshold=pot_threshold,
        capacitance=capacitance,
        time_constant=time_constant,
        tuning_function=TUNING_FUNCTION["gauss"],
        presentation_time=presentation_time,
        resolution_perlin=resolution_func_map,
        num_patches=num_patches,
        use_input_neurons=True if network_type == NETWORK_TYPE["input_only"] else False,
        img_prop=img_prop,
        spatial_sampling=spatial_sampling,
        use_dc=use_dc,
        save_plots=False,
        verbosity=1,
    )

    print_msg("Import network")
    # network.import_net()
    network.create_network()

    firing_rates, (spikes_s, time_s) = network.simulate(
        1000.,
        use_equilibrium=False
    )

    if use_single_neuron:
        network.img_prop = 1. / network.input_neurons_mask.sum()
        input_generator = network.set_input_rate(
            input_rate=max_firing_rate,
            origin=1000.,
            end=1000.,
            exc_only=True,
            tc=2
        )
    sim_time = 1000.
    spikes_s = None
    time_s = None
    for t in np.arange(1000., 2000., sim_time):
        if not use_single_neuron:
            nest.SetStatus(network.spike_gen, {"origin": t, "stop": 50.})
        firing_rates, (spikes_s, time_s) = network.simulate(
            sim_time,
            use_equilibrium=False
        )

        if use_single_neuron:
            network.set_input_generator(input_generator, input_rate=max_firing_rate, origin=t+sim_time, end=50.)

    recon = oblivious_stimulus_reconstruction(
        firing_rates,
        network.input_neurons_mask,
        network.ff_weight_mat,
        network.tuning_vector
    )
    plot_reconstruction(recon, recon, save_plots=save_plots, save_prefix=save_prefix)

    print_msg("Plot firing pattern over time")
    plot_spikes_over_time(
        spikes_s,
        time_s,
        network,
        title=network_name,
        t_start=0.,
        t_end=2000.,
        t_stim_start=np.arange(1000., 2000., sim_time),
        t_stim_end=np.arange(1050., 2000., sim_time),
        save_plot=save_plots,
        save_prefix=save_prefix
    )

    print_msg("Plot firing pattern over space")
    c_rgba = plot_spikes_over_space(
        firing_rates,
        network,
        title=network_name,
        save_plot=save_plots,
        save_prefix=save_prefix
    )

    print_msg("Create network animation")
    plot_network_animation(
        network,
        spikes_s,
        time_s,
        title=network_name,
        c_rgba=c_rgba,
        min_mem_pot=min_mem_pot,
        animation_start=1000.,
        animation_end=1500.,
        save_plot=save_plots,
        save_prefix=save_prefix
    )


if __name__ == '__main__':
    main()

