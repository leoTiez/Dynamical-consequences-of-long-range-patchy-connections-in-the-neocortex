#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.networkConstruction import *
from modules.createStimulus import *
from modules.networkAnalysis import *
from modules.thesisUtils import get_in_out_degree
from modules.networkParser import *
from modules.thesisConstants import *

from collections import Counter, OrderedDict
from scipy.spatial import KDTree
from pathlib import Path
import nest

nest.set_verbosity("M_ERROR")


class NeuronalNetworkBase:
    def __init__(
            self,
            num_sensory=int(1e4),
            ratio_inh_neurons=5,
            num_stim_discr=4,
            rec_factor=1.,
            ff_weight=1.,
            cap_s=1.,
            inh_weight=-5.,
            p_rf=0.7,
            rf_size=None,
            tuning_function=TUNING_FUNCTION["gauss"],
            global_connect=0.01,
            all_same_input_current=False,
            pot_threshold=-55.,
            pot_reset=-70.,
            capacitance=80.,
            time_constant=20.,
            layer_size=8.,
            sub_layer_size_ratio=0.8,
            max_spiking=1000.,
            bg_rate=500.,
            presentation_time=1000.,
            spacing_perlin=0.01,
            resolution_perlin=(15, 15),
            img_prop=1.,
            stimulus_size=(50, 50),
            network_type="local_circ_patchy_sd",
            spatial_sampling=False,
            num_spatial_samples=5,
            use_input_neurons=False,
            use_dc=False,
            verbosity=0,
            to_file=False,
            save_plots=False,
            save_prefix='',
            **kwargs
    ):
        """
        Neural network base class
        :param num_sensory: The number of sensory neurons
        :param ratio_inh_neurons: Every ratio_inh_neurons-th neuron is inhibitory,
        meaning that the ration is 1/ratio_inh_neurons
        :param num_stim_discr: The number of discriminated stimulus classes
        :param ff_weight: Weight of feedforward connections
        :param ff_factor: Factor by which the feedforward input is scaled. This is also the divisor of the recurrent
        weights and the maximal spiking rate for the Poisson spike generator
        :param cap_s: Excitatory weight
        :param inh_weight: Inhibitory weight
        :param p_rf: Connection probability of the receptive field
        :param rf_size: The size of the receptive field
        :param tuning_function: The tuning function, passed as an integer number defined in
        the dictionary TUNING_FUNCTION defined in the networkConstruction module
        :param all_same_input_current: Flag to determine whether all sensory neurons receive the same input.
        The flag mostly usful for bugfix and should be set to False otherwise
        :param pot_threshold: Threshold potential of the sensory neurons
        :param pot_reset: Reset potential of the sensory neurons
        :param capacitance: Capacitance of the sensory neurons
        :param time_constant: Time constant tau of the sensory neurons
        :param layer_size: Size of the sheet of the neural tissue that is modelled
        :param max_spiking: Maximal spiking rate of the ff input Poisson spike generator
        :param bg_rate: Rate that is used for background activity
        :param presentation_time: Time the image is presented to the network. Only possible if Poisson spike generator
        is used
        :param spacing_perlin: The space between two points in x and y for which an interpolation is computed. This
        value is used for creating the tuning map
        :param resolution_perlin: The resolution of the sampled values
        :param img_prop: Amount of information of the input image that is presented to the network
        :param network_type: The name of the network type as a string
        :param spatial_sampling: If true, the neurons that receive ff input are chosen with a spatial correlation
        :param num_spatial_samples: Determines the number of centers that are chosen for the spatial sampling
        :param use_input_neurons: If set to True, the reconstruction error based on input is used
        :param use_dc: Flag to determine whether to use a DC as injected current. If set to False a Poisson spike
        generator is used
        :param verbosity: Verbosity flag handles amount of output and created plot
        :param to_file: If set to true, the spikes are written to a file
        :param save_plots: Flag determines whether plots are saved or shown
        :param save_prefix: A saving prefix that can be used before every image to distinguish between different
        :param kwargs: Key work arguments that are not necessary
        experiments and trials
        """

        self.num_sensory = int(num_sensory)
        self.ratio_inh_neurons = ratio_inh_neurons
        self.num_stim_discr = num_stim_discr
        self.rec_factor = float(rec_factor)
        self.ff_weight = ff_weight
        self.cap_s = (1 - img_prop)**2 * cap_s * rec_factor
        self.inh_weight = (1 - img_prop)**2 * inh_weight * rec_factor
        self.p_rf = p_rf

        self.rf_size = rf_size
        if self.rf_size is None:
            self.rf_size = (stimulus_size[0] // 5, stimulus_size[1] // 5)

        self.all_same_input_current = all_same_input_current
        self.tuning_function = tuning_function
        self.global_connect = global_connect

        self.pot_threshold = pot_threshold
        self.pot_reset = pot_reset
        self.capacitance = capacitance
        self.time_constant = time_constant
        self.layer_size = layer_size
        self.sub_layer_size = self.layer_size * sub_layer_size_ratio
        self.max_spiking = max_spiking
        self.bg_rate = bg_rate
        self.presentation_time = presentation_time

        self.spacing_perlin = spacing_perlin
        self.resolution_perlin = resolution_perlin

        self.img_prop = img_prop
        self.stimulus_size = stimulus_size
        self.network_type = network_type
        self.spatial_sampling = spatial_sampling
        self.num_spatial_samples = num_spatial_samples

        self.use_input_neurons = use_input_neurons
        self.use_dc = use_dc

        self.verbosity = verbosity
        self.to_file = to_file
        self.save_plots = save_plots
        self.save_prefix = save_prefix

        self.plot_rf_relation = False if verbosity < 4 else True
        self.plot_tuning_map = False if verbosity < 4 else True

        self.torus_layer = None
        self.spike_detect = None
        self.multi_meter = None
        self.spike_gen = None
        self.torus_layer_tree = None
        self.torus_layer_nodes = None
        self.torus_inh_nodes = None
        self.torus_layer_positions = None
        self.input_neurons_mask = None
        self.input_inh_neurons_mask = None

        self.tuning_to_neuron_map = None
        self.tuning_vector = None
        self.color_map = None

        self.ff_weight_mat = None
        self.rf_list = None
        self.adj_sens_sens_mat = None
        self.rf_center_map = None
        self.input_recon = None

    # #################################################################################################################
    # Network creation
    # #################################################################################################################

    def create_layer(self):
        """
        Creates the neural sheet with inhibitory and excitatory neurons and creates the necessary
        class variables.
        :return: None
        """
        if self.verbosity > 0:
            print_msg("Create sensory layer")

        # Check ups
        if self.num_sensory < 1:
            raise ValueError("The number of sensory neuron is not set to a meaningful value."
                             " Set it larger than 1. Current value is %s" % self.num_sensory)
        if self.pot_threshold is None:
            raise ValueError("Threshold potential must not be None")
        if self.pot_reset is None:
            raise ValueError("Reset potential must not be None")
        if self.capacitance is None:
            raise ValueError("Capacitance must not be None")
        if self.layer_size <= 0:
            raise ValueError("The size of the layer is not set to a meaningful value."
                             " Set it larger than 0. Current value is %s" % self.layer_size)

        # Create the layer
        self.torus_layer, self.spike_detect, self.multi_meter, self.spike_gen = create_torus_layer_uniform(
            self.num_sensory,
            threshold_pot=self.pot_threshold,
            capacitance=self.capacitance,
            rest_pot=self.pot_reset,
            time_const=self.time_constant,
            size_layer=self.layer_size,
            bg_rate=self.bg_rate,
            p_rf=self.p_rf,
            synaptic_strength=self.ff_weight,
            to_torus=False,
            to_file=self.to_file
        )
        self.torus_layer_nodes = nest.GetNodes(self.torus_layer, properties={"element_type": "neuron"})[0]
        self.torus_layer_positions = tp.GetPosition(self.torus_layer_nodes)
        self.torus_layer_tree = KDTree(self.torus_layer_positions)
        self.torus_inh_nodes = np.random.choice(
            np.asarray(self.torus_layer_nodes),
            size=self.num_sensory // self.ratio_inh_neurons,
            replace=False
        ).tolist()

        sublayer_mask_specs = {
            "lower_left": [-self.sub_layer_size / 2., -self.sub_layer_size / 2.],
            "upper_right": [self.sub_layer_size / 2., self.sub_layer_size / 2.]
        }
        input_neurons = np.asarray(tp.SelectNodesByMask(
            self.torus_layer,
            (0., 0.),
            mask_obj=tp.CreateMask("rectangular", specs=sublayer_mask_specs)
        ))
        input_inh_neurons = np.asarray(list(set(input_neurons).intersection(set(self.torus_inh_nodes))))

        self.input_neurons_mask = np.zeros(self.num_sensory).astype("bool")
        self.input_neurons_mask[input_neurons - min(self.torus_layer_nodes)] = True
        self.input_inh_neurons_mask = np.zeros(self.num_sensory).astype("bool")
        self.input_inh_neurons_mask[input_inh_neurons - min(self.torus_inh_nodes)] = True

    def create_orientation_map(self):
        """
        Create the tuning map for the excitatory neurons. Inhibitory neurons won't show any tuning preference.
        :return: None
        """
        # Create stimulus tuning map
        if self.verbosity > 0:
            print_msg("Create stimulus tuning map")

        # Check ups
        if self.torus_layer is None:
            raise ValueError("The neural sheet has not been created yet. Run create_layer")
        if self.torus_inh_nodes is None:
            raise ValueError("The inhibitory nodes must be chosen first. Run create_layer")
        if self.num_stim_discr <= 0:
            raise ValueError("The number of stimulus feature classes is not set to a meaningful value."
                             " Set it larger than 0. Current value is %s" % self.num_stim_discr)
        if self.spacing_perlin <= 0:
            raise ValueError("The spacing of the interpolated values for the Perlin noise is not set "
                             "to a meaningful value. Set it larger than 0."
                             " Current value is %s" % self.spacing_perlin)
        if self.resolution_perlin[0] < 4 or self.resolution_perlin[1] < 4:
            raise ValueError("The resolution of the mesh for the Perlin noise is not set to a meaningful value."
                             " Set it larger equal than %s. Current value is %s"
                             % (4, self.resolution_perlin))

        (self.tuning_to_neuron_map,
         self.tuning_vector,
         self.color_map) = create_perlin_stimulus_map(
            self.torus_layer,
            self.torus_inh_nodes,
            num_stimulus_discr=self.num_stim_discr,
            plot=self.plot_tuning_map,
            spacing=self.spacing_perlin,
            resolution=self.resolution_perlin,
            save_plot=self.save_plots,
            save_prefix=self.save_prefix
        )

    def _choose_ff_neurons(self, exc_only=False, tc=None):
        num_input_neurons = int(self.img_prop * self.input_neurons_mask.sum())
        neuron_mask = np.zeros(self.num_sensory).astype("bool")
        neuron_mask |= self.input_neurons_mask
        if exc_only:
            neuron_mask &= ~self.input_inh_neurons_mask
        if tc is not None:
            tuning_mask = np.zeros(self.num_sensory).astype("bool")
            tuning_mask[np.asarray(self.tuning_to_neuron_map[tc]) - min(self.torus_layer_nodes)] = True
            neuron_mask &= tuning_mask

        if self.img_prop == 1.0:
            neurons_with_input = np.asarray(self.torus_layer_nodes)[neuron_mask]
        else:
            if not self.spatial_sampling:
                neurons_with_input_idx = np.random.choice(
                    np.arange(self.num_sensory).astype("int")[neuron_mask],
                    num_input_neurons,
                    replace=False
                ).tolist()

                neurons_with_input = np.asarray(self.torus_layer_nodes)[neurons_with_input_idx]
            else:
                if exc_only:
                    Warning("Not yet implemented to choose only excitatory neurons when using spatial sampling")
                sample_centers_idx = np.random.choice(
                    np.arange(self.num_sensory).astype("int")[neuron_mask],
                    self.num_spatial_samples,
                    replace=False
                )

                sample_centers = np.asarray(self.torus_layer_positions)[sample_centers_idx]
                k = int(num_input_neurons / self.num_spatial_samples)
                sublayer_tree = KDTree(np.asarray(self.torus_layer_positions)[self.input_neurons_mask])
                while True:
                    _, neurons_with_input_idx = sublayer_tree.query(
                        sample_centers,
                        k=k
                    )

                    neurons_with_input_idx = list(set(neurons_with_input_idx.flatten()))
                    diff = len(neurons_with_input_idx) - num_input_neurons
                    if diff > 0:
                        neurons_with_input_idx = neurons_with_input_idx[:num_input_neurons]
                        break
                    k += np.maximum(-diff, 1)

                chosen_neurons = np.zeros(self.num_sensory).astype("bool")
                chosen_neurons[
                    np.arange(self.num_sensory)[self.input_neurons_mask][neurons_with_input_idx]
                ] = True
                neurons_with_input = np.asarray(self.torus_layer_nodes)[np.logical_and(chosen_neurons, neuron_mask)]

        return neurons_with_input.tolist()

    def create_rf(self):
        if self.verbosity > 0:
            print_msg("Create receptive fields")

        # Check ups
        if self.layer_size <= 0:
            raise ValueError("The size of the layer is not set to a meaningful value."
                             " Set it larger than 0. Current value is %s" % self.layer_size)
        if self.stimulus_size[0] <= 0 or self.stimulus_size[1] <= 0:
            raise ValueError("Stimulus size must be greater than 0")
        if self.torus_layer is None:
            raise ValueError("The neural sheet has not been created yet. Run create_layer")
        if self.torus_inh_nodes is None:
            raise ValueError("The inhibitory nodes must be chosen first. Run create_layer")
        if self.p_rf < 0 or self.p_rf > 1:
            raise ValueError("The the connection probability for the receptive field is not set "
                             "to a meaningful value. Set it between 0 and 1"
                             " Current value is %s" % self.p_rf)
        if self.rf_size[0] < 0 or self.rf_size[1] < 0:
            raise ValueError("The size and shape of the receptive field must not be negative.")

        self.rf_center_map = [
            (
                (y + (self.sub_layer_size / 2.)) / float(self.sub_layer_size) * self.stimulus_size[1],
                (x + (self.sub_layer_size / 2.)) / float(self.sub_layer_size) * self.stimulus_size[0]
            )
            for (x, y) in np.asarray(self.torus_layer_positions)[self.input_neurons_mask]
        ]

        self.rf_list = create_rf(self, self.stimulus_size)

    def create_retina(self, input_stimulus):
        """
        Creates the receptive fields and computes the injected DC / spike rate of a Poisson generator for every
        sensory neuron.
        :return: None
        """
        if self.verbosity > 0:
            print_msg("Create central points for receptive fields")

        # Check ups
        if self.layer_size <= 0:
            raise ValueError("The size of the layer is not set to a meaningful value."
                             " Set it larger than 0. Current value is %s" % self.layer_size)
        if self.torus_layer is None:
            raise ValueError("The neural sheet has not been created yet. Run create_layer")
        if self.torus_inh_nodes is None:
            raise ValueError("The inhibitory nodes must be chosen first. Run create_layer")
        if self.tuning_vector is None:
            raise ValueError("The orientation map must be created first. Run create_orientation_map")
        if self.ff_weight is None:
            raise ValueError("The feedforward weight must not be None. Run determine_ffweight")
        if self.p_rf < 0 or self.p_rf > 1:
            raise ValueError("The the connection probability for the receptive field is not set "
                             "to a meaningful value. Set it between 0 and 1"
                             " Current value is %s" % self.p_rf)
        if self.rf_size[0] < 0 or self.rf_size[1] < 0:
            raise ValueError("The size and shape of the receptive field must not be negative.")
        if self.ff_weight_mat is None:
            raise ValueError("The feedforward adjacency matrix must not be None. Run create_rf.")
        # Create connections to receptive field
        if self.verbosity > 0:
            print_msg("Calculate injected stimulus")

        neurons_with_input = self._choose_ff_neurons()

        if self.plot_tuning_map:
            inh_mask = np.zeros(self.num_sensory).astype('bool')
            inh_mask[np.asarray(self.torus_inh_nodes) - min(self.torus_layer_nodes)] = True

            x_grid, y_grid = coordinates_to_cmap_index(
                self.layer_size,
                np.asarray(self.torus_layer_positions)[~inh_mask],
                self.spacing_perlin
            )
            stim_class = self.color_map[x_grid, y_grid]
            muted_nodes = list(set(self.torus_layer_nodes).difference(set(neurons_with_input)))
            plot_cmap(
                ff_nodes=neurons_with_input,
                inh_nodes=self.torus_inh_nodes,
                color_map=self.color_map,
                stim_class=stim_class,
                positions=self.torus_layer_positions,
                muted_nodes=muted_nodes,
                size_layer=self.layer_size,
                resolution=self.resolution_perlin,
                num_stimulus_discr=self.num_stim_discr,
                save_plot=self.save_plots,
                plot_sublayer=True,
                save_prefix=self.save_prefix
            )

        self.input_recon = create_connections_rf(
            input_stimulus,
            neurons_with_input,
            self,
            rf_list=self.rf_list
        )

    # #################################################################################################################
    # Simulate
    # #################################################################################################################

    def simulate(self, simulation_time=250., use_equilibrium=False, eq_time=600.):
        """
        Simulate the network
        :param simulation_time: The simulation time in milliseconds
        :param use_equilibrium: If set to true, only the last eq_time ms are used to compute the average firing rate, ie
        when the network is expected to approach equilibrium
        :param eq_time: The time after which the network is assumed to reach equilibrium
        :return: The firing rates, (node IDs of the spiking neurons, the respective spike times)
        """
        if self.verbosity > 0:
            print_msg("Simulate")

        # Check ups
        if self.spike_detect is None:
            raise ValueError("The spike detector must not be None. Run create_layer")

        nest.Simulate(float(simulation_time))

        # Get network response in spikes
        data_sp = nest.GetStatus(self.spike_detect, keys="events")[0]
        spikes_s = np.asarray(data_sp["senders"])
        time_s = np.asarray(data_sp["times"])
        global_time = nest.GetKernelStatus("time")

        firing_rates = get_firing_rates(
            spikes_s[time_s > global_time - simulation_time] if not use_equilibrium
            else spikes_s[time_s > global_time - simulation_time + eq_time],
            self.torus_layer_nodes,
            simulation_time if not use_equilibrium else simulation_time - eq_time
        )
        return firing_rates, (spikes_s, time_s)

    # #################################################################################################################
    # Plot connections
    # #################################################################################################################

    def plot_connections_node(self, node_idx=0, plot_name="all_connections.png"):
        """
        Plotting function to plot all connections of a particular node
        :param node_idx: The index of the node in the list of neurons (note that this is not the node ID)
        :param plot_name: Name of the plot if self.save_plots is set to True
        :return: None
        """
        # Check ups
        if self.torus_layer_nodes is None:
            raise ValueError("The sensory nodes have not been created yet. Run create_layer")
        if self.layer_size <= 0:
            raise ValueError("The size of the layer is not set to a meaningful value."
                             " Set it larger than 0. Current value is %s" % self.layer_size)
        connect = nest.GetConnections([self.torus_layer_nodes[node_idx]])
        targets = nest.GetStatus(connect, "target")
        sensory_targets = [t for t in targets if t in list(self.torus_layer_nodes)]
        plot_connections(
            [self.torus_layer_nodes[node_idx]],
            sensory_targets,
            self.layer_size,
            save_plot=self.save_plots,
            plot_name=plot_name,
            save_prefix=self.save_prefix,
            color_mask=self.color_map
        )

    def _connect_distribution(self, plot_local=False, plot_lr=False, plot_name="in_out_deg_dist.png", **kwargs):
        """
        Plot the in-/outdegree distribution in the network
        :param plot_name: Name of the plot
        :return: None
        """
        # Check ups
        if self.torus_layer_nodes is None:
            raise ValueError("The sensory nodes have not been created yet. Run create_layer")

        tree = KDTree(np.asarray(self.torus_layer_positions)[self.input_neurons_mask])
        in_degree, out_degree, in_degree_loc, out_degree_loc, in_degree_lr, out_degree_lr = get_in_out_degree(
            np.asarray(self.torus_layer_nodes)[self.input_neurons_mask],
            node_tree=tree if plot_local or plot_lr else None,
            node_pos=np.asarray(self.torus_layer_positions)[self.input_neurons_mask] if plot_local or plot_lr else None,
            r_loc=self.r_loc if plot_local or plot_lr else None,
            size_layer=self.layer_size
        )

        in_deg_dist = OrderedDict(sorted(Counter(in_degree).items()))
        out_deg_dist = OrderedDict(sorted(Counter(out_degree).items()))

        num_rows = 1
        if plot_local:
            num_rows += 1
        if plot_lr:
            num_rows += 1

        fig, ax = plt.subplots(num_rows, 2, figsize=(10, 10))

        if not plot_local and not plot_lr:
            ax[0].bar(list(in_deg_dist.keys()), list(in_deg_dist.values()))
            ax[0].set_ylabel("Total")
            ax[1].bar(list(out_deg_dist.keys()), list(out_deg_dist.values()))
            ax[0].set_xlabel("Indegree")
            ax[1].set_xlabel("Outdegree")

        else:
            ax[0][0].bar(list(in_deg_dist.keys()), list(in_deg_dist.values()))
            ax[0][0].set_ylabel("Total")
            ax[0][1].bar(list(out_deg_dist.keys()), list(out_deg_dist.values()))

            if plot_local:
                in_deg_dist_loc = OrderedDict(sorted(Counter(in_degree_loc).items()))
                out_deg_dist_loc = OrderedDict(sorted(Counter(out_degree_loc).items()))

                ax[1][0].bar(list(in_deg_dist_loc.keys()), list(in_deg_dist_loc.values()))
                ax[1][0].set_ylabel("Proximal")
                ax[1][1].bar(list(out_deg_dist_loc.keys()), list(out_deg_dist_loc.values()))

            if plot_lr:
                in_deg_dist_lr = OrderedDict(sorted(Counter(in_degree_lr).items()))
                out_deg_dist_lr = OrderedDict(sorted(Counter(out_degree_lr).items()))

                ax[2][0].bar(list(in_deg_dist_lr.keys()), list(in_deg_dist_lr.values()))
                ax[2][0].set_ylabel("Distal")
                ax[2][1].bar(list(out_deg_dist_lr.keys()), list(out_deg_dist_lr.values()))

            ax[-1][0].set_xlabel("Indegree")
            ax[-1][1].set_xlabel("Outdegree")

        fig.text(0.03, 0.5, "#Nodes", ha="center", va="center", rotation="vertical")

        if self.save_plots:
            curr_dir = os.getcwd()
            Path(curr_dir + "/figures/in-out-dist/").mkdir(parents=True, exist_ok=True)
            plt.savefig(curr_dir + "/figures/in-out-dist/%s_%s" % (self.save_prefix, plot_name))
        else:
            plt.show()

    # #################################################################################################################
    # Getter / Setter
    # #################################################################################################################

    def get_sensory_weight_mat(self):
        """
        Getter function for the sensory weight matrix
        :return: Sensory weight matrix
        """
        if self.verbosity > 0:
            print_msg("Create adjacency matrix for sensory-to-sensory connections")
        if self.adj_sens_sens_mat is None:
            self.adj_sens_sens_mat = create_adjacency_matrix(self.torus_layer_nodes, self.torus_layer_nodes)
        return self.adj_sens_sens_mat

    def set_recurrent_weight(self, weight, divide_by_num_connect=False):
        """
        Setter function for all recurrent excitatory weights. This function becomes hand if this value needs to be
        dynamically or is dependent on the number of established connections in the network.
        :param weight: Weight value
        :param divide_by_num_connect: If set to true the passed weight is divided by the number of connections in
        the network
        :return: None
        """
        if self.verbosity > 0:
            print_msg("Set synaptic weights for sensory to sensory neurons")
        adj_sens_sens_mat = self.get_sensory_weight_mat()
        set_synaptic_strength(
            self.torus_layer_nodes,
            adj_sens_sens_mat,
            cap_s=weight,
            divide_by_num_connect=divide_by_num_connect
        )

    def set_input_generator(self, input_generators, input_rate=None, origin=0., start=0., end=1000.):
        if self.use_dc:
            Warning("Cannot set input rate when using DC input. Nothing changed.")
            return

        if input_rate is None:
            input_rate = self.max_spiking

        nest.SetStatus(input_generators, {"rate": input_rate, "origin": origin, "start": start, "stop": end})

    def set_input_rate(self, input_rate=None, origin=0., start=0., end=1000., exc_only=True, tc=None):
        if self.use_dc:
            Warning("Cannot set input rate when using DC input. Nothing changed.")
            return

        input_neurons = self._choose_ff_neurons(exc_only=exc_only, tc=tc)
        input_generators = np.asarray(self.spike_gen)[np.asarray(input_neurons) - min(self.torus_layer_nodes)].tolist()
        self.set_input_generator(
            input_generators=input_generators,
            input_rate=input_rate,
            origin=origin,
            start=start,
            end=end
        )

        return input_generators

    # #################################################################################################################
    # Abstract methods
    # #################################################################################################################

    def _set_nest_kernel(self):
        """
        Reset the nest kernel. It is triggered when creating or loading a network
        :return:
        """
        nest.ResetKernel()
        curr_dir = os.getcwd()
        path = "%s/network_files/spikes/" % curr_dir
        Path(path).mkdir(parents=True, exist_ok=True)
        nest.SetKernelStatus({
            "overwrite_files": True,
            "data_path": path,
            "data_prefix": self.save_prefix
        })

    def create_network(self, input_stimulus=None):
        """
        Creates the network and sets up all necessary connections
        :return: None
        """
        # Reset Nest Kernel
        self._set_nest_kernel()

        self.create_layer()
        self.create_orientation_map()
        self.create_rf()
        if not self.all_same_input_current and input_stimulus is not None:
            self.create_retina(input_stimulus)

    def export_net(self, feature_folder=""):
        """
        Export the network neuron positions and connections
        :param feature_folder: If a the network should be saved to a particular feature folder
        :return: None
        """
        save_net(self, self.network_type, feature_folder=feature_folder)

    def import_net(self, input_stimulus=None):
        """
        Loading a network from file. This resets the current nest kernel
        :return: None
        """
        self._set_nest_kernel()
        load_net(self, self.network_type)
        if input_stimulus is not None:
            self.create_retina(input_stimulus)


class RandomNetwork(NeuronalNetworkBase):
    def __init__(
            self,
            num_sensory=int(1e4),
            layer_size=8.,
            verbosity=0,
            **kwargs
    ):
        """
        Random network class
        :param num_sensory: Number of sensory nodes in the sheet
        :param layer_size: Size of the layer
        :param r_loc: Defines the radius for local connections. Although this parameter is not used for establishing
        any distance dependent functions, it is applied for determining a connection probability that makes a comparison
        possible between the different network types
        :param verbosity: Verbosity flag to determine the amount of printed output and created plots
        :param kwargs: Key value arguments that are passed to the base class
        """
        spacing_perlin = layer_size / np.sqrt(num_sensory)
        res_perlin = int(layer_size * np.sqrt(num_sensory))
        resolution_perlin = (res_perlin, res_perlin)
        self.__dict__.update(kwargs)
        kwargs.pop("resolution_perlin", None)
        kwargs.pop("spacing_perlin", None)
        NeuronalNetworkBase.__init__(
            self,
            num_sensory=num_sensory,
            layer_size=layer_size,
            spacing_perlin=spacing_perlin,
            resolution_perlin=resolution_perlin,
            verbosity=verbosity,
            **kwargs
        )
        # Set probability to a value such that it becomes comparable to clustered networks
        # Calculation is taken from Voges et al.
        self.p_random = self.global_connect

        self.plot_random_connections = False if verbosity < 4 else True

    def create_random_connections(self):
        """
        Establish random connections to other nodes in the network
        :return: None
        """
        if self.verbosity > 0:
            print_msg("Create random connections")

        # Check ups
        if self.torus_layer is None:
            raise ValueError("The neural sheet has not been created yet. Run create_layer")
        if self.torus_inh_nodes is None:
            raise ValueError("The inhibitory nodes must be chosen first. Run create_layer")
        if self.inh_weight is None:
            raise ValueError("The inhibitory weight must not be None")
        if self.cap_s < 0:
            raise ValueError("Lateral synaptic weight must not be negative")
        if self.p_random < 0 or self.p_random > 1:
            raise ValueError("The the connection probability for the random connections is not set "
                             "to a meaningful value. Set it between 0 and 1"
                             " Current value is %s" % self.p_random)

        connect_dict = {"rule": "pairwise_bernoulli", "p": self.p_random}
        create_random_connections(
            self.torus_layer,
            self.torus_inh_nodes,
            inh_weight=self.inh_weight,
            connect_dict=connect_dict,
            cap_s=self.cap_s,
            plot=self.plot_random_connections,
            save_plot=self.save_plots,
            save_prefix=self.save_prefix,
            color_mask=self.color_map
        )

    def connect_distribution(self, plot_name="in_out_deg_dist.png"):
        NeuronalNetworkBase._connect_distribution(self, plot_local=False, plot_lr=False, plot_name=plot_name)

    def create_network(self, input_stimulus=None):
        """
        Create the network and establish the connections. Calls create function of parent class
        :return: None
        """
        NeuronalNetworkBase.create_network(self, input_stimulus)
        if self.use_input_neurons:
            return
        self.create_random_connections()


class LocalNetwork(NeuronalNetworkBase):
    ACCEPTED_LOC_CONN = ["circular", "sd"]

    def __init__(
            self,
            r_loc=0.5,
            c_alpha=1.,
            loc_connection_type="circular",
            verbosity=0,
            **kwargs
    ):
        """
        Class that establishes local connections with locally clustered tuning specfic neurons
        :param c_alpha: Connection probability to connect to another neuron within the local radius
        :param r_loc: Radius within which a local connection is established
        :param loc_connection_type: Connection policy for local connections. This can be any value in the
        ACCEPTED_LOC_CONN list. Circ are circular connections, whereas sd are stimulus dependent connections
        :param verbosity: Determines the amount of output and created plots
        :param kwargs: Arguments that are passed to parent class
        """
        self.__dict__.update(kwargs)
        NeuronalNetworkBase.__init__(
            self,
            verbosity=verbosity,
            **kwargs
        )

        self.r_loc = r_loc
        if c_alpha > 1.0 or c_alpha < 0.0:
            raise ValueError("c_alpha must be set between 0 and 1")
        self.c_alpha = c_alpha

        self.p_loc = self.global_connect * self.c_alpha * self.layer_size**2 / (np.pi * self.r_loc**2)

        self.loc_connection_type = loc_connection_type.lower()
        if self.loc_connection_type not in LocalNetwork.ACCEPTED_LOC_CONN:
            raise ValueError("The passed connection type %s is not accepted." % self.loc_connection_type)

        self.plot_local_connections = False if verbosity < 4 else True

    def create_local_connections(self):
        """
        Create local connections in the neural sheet
        :return: None
        """
        # Check up
        if self.torus_layer is None:
            raise ValueError("The neural sheet has not been created yet. Run create_layer")
        if self.torus_inh_nodes is None:
            raise ValueError("The inhibitory nodes must be chosen first. Run create_layer")
        if self.inh_weight is None:
            raise ValueError("The inhibitory weight must not be None")
        if self.cap_s < 0:
            raise ValueError("Lateral synaptic weight must not be negative")
        if self.p_loc < 0 or self.p_loc > 1:
            raise ValueError("The the connection probability for the local connections is not set "
                             "to a meaningful value. Set it between 0 and 1"
                             " Current value is %s" % self.p_loc)
        if self.r_loc < 0:
            raise ValueError("Local connection radius must not be negative")
        if self.torus_layer_tree is None:
            raise ValueError("The torus layer organised in a tree must be created first. Run create_layer")

        # Set connection dict
        local_connect_dict = {"rule": "pairwise_bernoulli", "p": self.p_loc}

        if self.loc_connection_type == "sd":
            if self.verbosity > 0:
                print_msg("Create local stimulus dependent connections")
            # Connection specific check up
            if self.tuning_vector is None:
                raise ValueError("The mapping from neuron to orientation tuning must be created first."
                                 " Run create_orientation_map")
            if self.tuning_to_neuron_map is None:
                raise ValueError("The mapping from orientation tuning to neuron must be created first."
                                 " Run create_orientation_map")

            create_stimulus_based_local_connections(
                self.torus_layer,
                self.torus_layer_tree,
                self.tuning_vector,
                self.tuning_to_neuron_map,
                self.torus_inh_nodes,
                inh_weight=self.inh_weight,
                cap_s=self.cap_s,
                connect_dict=local_connect_dict,
                r_loc=self.r_loc,
                plot=self.plot_local_connections,
                color_mask=self.color_map,
                save_plot=self.save_plots,
                save_prefix=self.save_prefix
            )
        elif self.loc_connection_type == "circular":
            if self.verbosity > 0:
                print_msg("Create local circular connections")
            create_local_circular_connections(
                self.torus_layer,
                self.torus_layer_tree,
                self.torus_inh_nodes,
                inh_weight=self.inh_weight,
                connect_dict=local_connect_dict,
                r_loc=self.r_loc,
                cap_s=self.cap_s,
                plot=self.plot_local_connections,
                color_mask=self.color_map,
                save_plot=self.save_plots,
                save_prefix=self.save_prefix
            )
        else:
            raise ValueError("The passed connection type %s is not accepted." % self.loc_connection_type)

    def connect_distribution(self, plot_name="in_out_deg_dist.png"):
        """
        Create plot for the in/outdegree distribution. There are different plots created for all connections and
        for only local connections
        :param plot_name: The name of the plot if the self.save_plots flag is set to true
        :return: None
        """
        NeuronalNetworkBase._connect_distribution(self, plot_local=True, plot_lr=False, plot_name=plot_name)

    def create_network(self, input_stimulus=None):
        """
        Creates the network and class the create function of the parent class
        :return: None
        """
        NeuronalNetworkBase.create_network(self, input_stimulus)
        if self.use_input_neurons:
            return
        self.create_local_connections()


class PatchyNetwork(LocalNetwork):
    ACCEPTED_LR_CONN = ["random", "sd"]

    def __init__(
            self,
            c_alpha=0.7,
            num_patches=3,
            lr_connection_type="sd",
            verbosity=0,
            **kwargs
    ):
        """
        Class for patchy networks with long-range patchy connections
        :param p_lr: Connection probability for long-range patchy connections
        :param num_patches: Number of patches per neuron
        :param lr_connection_type: The connection type of the long-range patchy connections. Can be any value from the
        ACCEPTED_LR_CONN list. Random patches can be created everywhere within in given distance, whereas sd establishes
        only connections to neurons with the same stimulus preference
        :param verbosity: Flag to determine the amount of output and created plots
        :param kwargs: The key value pairs that are passed to the parent class
        """

        self.__dict__.update(kwargs)
        LocalNetwork.__init__(
            self,
            c_alpha=c_alpha,
            verbosity=verbosity,
            **kwargs
        )

        self.num_patches = num_patches
        self.r_p = self.r_loc / 2.

        self.p_lr = self.global_connect * (1 - self.c_alpha) * self.layer_size**2 \
                    / (self.num_patches * np.pi * self.r_p**2)
        self.lr_connection_type = lr_connection_type.lower()
        if self.lr_connection_type not in PatchyNetwork.ACCEPTED_LR_CONN:
            raise ValueError("%s is not an accepted long-range connection type" % self.lr_connection_type)

        self.plot_patchy_connections = False if verbosity < 4 else True

    def create_lr_connections(self):
        """
        Create long-range connections
        :return: None
        """
        # Check up
        if self.torus_layer is None:
            raise ValueError("The neural sheet has not been created yet. Run create_layer")
        if self.torus_inh_nodes is None:
            raise ValueError("The inhibitory nodes must be chosen first. Run create_layer")
        if self.inh_weight is None:
            raise ValueError("The inhibitory weight must not be None")
        if self.cap_s < 0:
            raise ValueError("Lateral synaptic weight must not be negative")
        if self.p_loc < 0 or self.p_loc > 1:
            raise ValueError("The the connection probability for the local connections is not set "
                             "to a meaningful value. Set it between 0 and 1"
                             " Current value is %s" % self.p_loc)
        if self.p_lr < 0 or self.p_lr > 1:
            raise ValueError("The the connection probability for the long-range connections is not set "
                             "to a meaningful value. Set it between 0 and 1"
                             " Current value is %s" % self.p_lr)
        if self.r_loc < 0:
            raise ValueError("Local connection radius must not be negative")
        if self.num_patches < 0:
            raise ValueError("Number of patches must not be negative")

        if self.lr_connection_type == "sd":
            if self.verbosity > 0:
                print_msg("Create long-range patchy stimulus dependent connections")
            # Connection specific check up
            if self.torus_layer_tree is None:
                raise ValueError("The torus layer organised in a tree must be created first. Run create_layer")
            if self.tuning_vector is None:
                raise ValueError("The mapping from neuron to orientation tuning must be created first."
                                 " Run create_orientation_map")
            if self.tuning_to_neuron_map is None:
                raise ValueError("The mapping from orientation tuning to neuron must be created first."
                                 " Run create_orientation_map")

            patchy_connect_dict = {"rule": "pairwise_bernoulli", "p": self.p_lr}
            create_stimulus_based_patches_random(
                self.torus_layer,
                self.tuning_vector,
                self.tuning_to_neuron_map,
                self.torus_inh_nodes,
                self.torus_layer_tree,
                r_loc=self.r_loc,
                r_p=self.r_p,
                cap_s=self.cap_s,
                connect_dict=patchy_connect_dict,
                num_patches=self.num_patches,
                plot=self.plot_patchy_connections,
                save_plot=self.save_plots,
                save_prefix=self.save_prefix,
                color_mask=self.color_map
            )

        if self.lr_connection_type == "random":
            if self.verbosity > 0:
                print_msg("Create long-range patchy random connections")
            create_random_patches(
                self.torus_layer,
                self.torus_inh_nodes,
                r_loc=self.r_loc,
                p_loc=self.p_loc,
                cap_s=self.cap_s,
                r_p=self.r_p,
                p_p=self.p_lr,
                num_patches=self.num_patches,
                plot=self.plot_patchy_connections,
                save_plot=self.save_plots,
                save_prefix=self.save_prefix,
                color_mask=self.color_map
            )

    def connect_distribution(self, plot_name="in_out_deg_dist.png"):
        """
        Plot in/outdegree distribution. There are different subplots created for all connections, local, and long-range
        :param plot_name: Name of the plot if self.save_plots is set to True
        :return: None
        """
        NeuronalNetworkBase._connect_distribution(self, plot_local=True, plot_lr=True, plot_name=plot_name)

    def create_network(self, input_stimulus=None):
        """
        Create the network, establishes the connections and calls the create function of the parent class
        :return:
        """
        LocalNetwork.create_network(self, input_stimulus)
        if self.use_input_neurons:
            return
        self.create_lr_connections()


def network_factory(network_type=NETWORK_TYPE["local_circ_patchy_sd"], **kwargs):
    """
    Factory function to instantiate an neuronal network object
    :param network_type: The network type. The value can be any integer defined in the NETWORK_TYPE dictionary
    :param kwargs: The parameters passed to the network
    :return: The network object
    """
    if network_type == NETWORK_TYPE["random"]:
        network = RandomNetwork(
            network_type="random",
            **kwargs
        )
    elif network_type == NETWORK_TYPE["local_circ"]:
        kwargs.pop("c_alpha", None)
        network = LocalNetwork(
            c_alpha=1.,
            loc_connection_type="circular",
            network_type="local_circ",
            **kwargs
        )
    elif network_type == NETWORK_TYPE["local_sd"]:
        kwargs.pop("c_alpha", None)
        network = LocalNetwork(
            c_alpha=1.,
            loc_connection_type="sd",
            network_type="local_sd",
            **kwargs
        )
    elif network_type == NETWORK_TYPE["local_circ_patchy_random"]:
        network = PatchyNetwork(
            loc_connection_type="circular",
            lr_connection_type="random",
            network_type="local_circ_patchy_random",
            **kwargs
        )
    elif network_type == NETWORK_TYPE["local_circ_patchy_sd"]:
        network = PatchyNetwork(
            loc_connection_type="circular",
            lr_connection_type="sd",
            network_type="local_circ_patchy_sd",
            **kwargs
        )
    elif network_type == NETWORK_TYPE["local_sd_patchy_sd"]:
        network = PatchyNetwork(
            loc_connection_type="sd",
            lr_connection_type="sd",
            network_type="local_sd_patchy_sd",
            **kwargs
        )
    elif network_type == NETWORK_TYPE["input_only"]:
        network = NeuronalNetworkBase(
            network_type="input_stimulus",
            **kwargs
        )
    else:
        raise ValueError("Network type %s is not accepted" % network_type)

    return network

