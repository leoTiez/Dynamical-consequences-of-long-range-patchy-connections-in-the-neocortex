#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.networkConstruction import *
from modules.createStimulus import *
from modules.networkAnalysis import *
from modules.thesisUtils import get_in_out_degree

from collections import Counter, OrderedDict
from scipy.spatial import KDTree
from pathlib import Path
import nest

nest.set_verbosity("M_ERROR")

NETWORK_TYPE = {
    "random": 0,
    "local_circ": 1,
    "local_sd": 2,
    "local_circ_patchy_sd": 3,
    "local_circ_patchy_random": 4,
    "local_sd_patchy_sd": 5,
    "input_only": 6
}


class NeuronalNetworkBase:
    def __init__(
            self,
            input_stimulus,
            num_sensory=int(1e4),
            ratio_inh_neurons=5,
            num_stim_discr=4,
            ff_weight=1.,
            cap_s=1.,
            inh_weight=-15.,
            p_rf=0.3,
            rf_size=None,
            tuning_function=TUNING_FUNCTION["step"],
            all_same_input_current=False,
            pot_threshold=-55.,
            pot_reset=-70.,
            capacitance=80.,
            time_constant=20.,
            layer_size=8.,
            spacing_perlin=0.01,
            resolution_perlin=(15, 15),
            img_prop=1.,
            spatial_sampling=False,
            num_spatial_samples=5,
            use_input_neurons=False,
            use_dc=True,
            verbosity=0,
            save_plots=False,
            save_prefix='',
            **kwargs
    ):
        """
        Neural network base class
        :param input_stimulus: The input image
        :param num_sensory: The number of sensory neurons
        :param ratio_inh_neurons: Every ratio_inh_neurons-th neuron is inhibitory,
        meaning that the ration is 1/ratio_inh_neurons
        :param num_stim_discr: The number of discriminated stimulus classes
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
        :param spacing_perlin: The space between two points in x and y for which an interpolation is computed. This
        value is used for creating the tuning map
        :param resolution_perlin: The resolution of the sampled values
        :param img_prop: Amount of information of the input image that is presented to the network
        :param spatial_sampling: If true, the neurons that receive ff input are chosen with a spatial correlation
        :param use_input_neurons: If set to True, the reconstruction error based on input is used
        :param use_dc: Flag to determine whether to use a DC as injected current. If set to False a Poisson spike
        generator is used
        :param verbosity: Verbosity flag handles amount of output and created plot
        :param save_plots: Flag determines whether plots are saved or shown
        :param save_prefix: A saving prefix that can be used before every image to distinguish between different
        :param kwargs: Key work arguments that are not necessary
        experiments and trials
        """

        self.input_stimulus = input_stimulus
        self.num_sensory = int(num_sensory)
        self.ratio_inh_neurons = ratio_inh_neurons
        self.num_stim_discr = num_stim_discr
        self.ff_weight = ff_weight
        self.cap_s = cap_s
        self.inh_weight = inh_weight
        self.p_rf = p_rf
        self.rf_size = rf_size

        if self.rf_size is None:
            self.rf_size = (input_stimulus.shape[0] // 4, input_stimulus.shape[1] // 4)

        self.all_same_input_current = all_same_input_current
        self.tuning_function = tuning_function

        self.pot_threshold = pot_threshold
        self.pot_reset = pot_reset
        self.capacitance = capacitance
        self.time_constant = time_constant
        self.layer_size = layer_size

        self.spacing_perlin = spacing_perlin
        self.resolution_perlin = resolution_perlin

        self.img_prop = img_prop
        self.spatial_sampling = spatial_sampling
        self.num_spatial_samples = num_spatial_samples

        self.use_input_neurons = use_input_neurons
        self.use_dc = use_dc

        self.verbosity = verbosity
        self.save_plots = save_plots
        self.save_prefix = save_prefix

        self.plot_rf_relation = False if verbosity < 4 else True
        self.plot_tuning_map = False if verbosity < 4 else True

        self.torus_layer = None
        self.spike_detect = None
        self.torus_layer_tree = None
        self.torus_layer_nodes = None
        self.torus_inh_nodes = None
        self.torus_layer_positions = None

        self.tuning_to_neuron_map = None
        self.neuron_to_tuning_map = None
        self.color_map = None

        self.ff_weight_mat = None
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
            print("\n#####################\tCreate sensory layer")

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
        self.torus_layer, self.spike_detect, _ = create_torus_layer_uniform(
            self.num_sensory,
            threshold_pot=self.pot_threshold,
            capacitance=self.capacitance,
            rest_pot=self.pot_reset,
            time_const=self.time_constant,
            size_layer=self.layer_size
        )
        self.torus_layer_nodes = nest.GetNodes(self.torus_layer, properties={"element_type": "neuron"})[0]
        self.torus_layer_positions = tp.GetPosition(self.torus_layer_nodes)
        self.torus_layer_tree = KDTree(self.torus_layer_positions)
        self.torus_inh_nodes = np.random.choice(
            np.asarray(self.torus_layer_nodes),
            size=self.num_sensory // self.ratio_inh_neurons,
            replace=False
        ).tolist()

    def create_orientation_map(self):
        """
        Create the tuning map for the excitatory neurons. Inhibitory neurons won't show any tuning preference.
        :return: None
        """
        # Create stimulus tuning map
        if self.verbosity > 0:
            print("\n#####################\tCreate stimulus tuning map")

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
         self.neuron_to_tuning_map,
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

    def create_retina(self):
        """
        Creates the receptive fields and computes the injected DC / spike rate of a Poisson generator for every
        sensory neuron.
        :return: None
        """
        if self.verbosity > 0:
            print("\n#####################\tCreate central points for receptive fields")

        # Check ups
        if self.layer_size <= 0:
            raise ValueError("The size of the layer is not set to a meaningful value."
                             " Set it larger than 0. Current value is %s" % self.layer_size)
        if self.input_stimulus is None:
            raise ValueError("The input stimulus must not be None")
        if self.torus_layer is None:
            raise ValueError("The neural sheet has not been created yet. Run create_layer")
        if self.torus_inh_nodes is None:
            raise ValueError("The inhibitory nodes must be chosen first. Run create_layer")
        if self.neuron_to_tuning_map is None:
            raise ValueError("The orientation map must be created first. Run create_orientation_map")
        if self.ff_weight is None:
            raise ValueError("The feedforward weight must not be None. Run determine_ffweight")
        if self.p_rf < 0 or self.p_rf > 1:
            raise ValueError("The the connection probability for the receptive field is not set "
                             "to a meaningful value. Set it between 0 and 1"
                             " Current value is %s" % self.p_rf)
        if self.rf_size[0] < 0 or self.rf_size[1] < 0:
            raise ValueError("The size and shape of the receptive field must not be negative.")

        # Create connections to receptive field
        if self.verbosity > 0:
            print("\n#####################\tCreate connections between receptors and sensory neurons")

        if self.img_prop == 1.0:
            neurons_with_input = np.asarray(self.torus_layer_nodes)[:]
            positions_with_input = np.asarray(self.torus_layer_positions)[:]
        else:
            if not self.spatial_sampling:
                neurons_with_input_idx = np.random.choice(
                    len(self.torus_layer_nodes),
                    int(self.img_prop * self.num_sensory),
                    replace=False
                ).tolist()
                neurons_with_input = np.asarray(self.torus_layer_nodes)[neurons_with_input_idx]
                positions_with_input = np.asarray(self.torus_layer_positions)[neurons_with_input_idx]

            else:
                sample_centers_idx = np.random.choice(
                    len(self.torus_layer_positions),
                    self.num_spatial_samples,
                    replace=False
                )
                sample_centers = np.asarray(self.torus_layer_positions)[sample_centers_idx]
                k = int(self.num_sensory / self.num_spatial_samples)
                while True:
                    _, neurons_with_input_idx = self.torus_layer_tree.query(
                        sample_centers,
                        k=k
                    )

                    neurons_with_input_idx = list(set(neurons_with_input_idx.flatten()))
                    diff = len(neurons_with_input_idx) - int(self.img_prop * self.num_sensory)
                    if diff > 0:
                        neurons_with_input_idx = neurons_with_input_idx[:int(self.img_prop * self.num_sensory)]
                        break
                    k += np.maximum(-diff, 1)

                neurons_with_input = np.asarray(self.torus_layer_nodes)[neurons_with_input_idx]
                positions_with_input = np.asarray(self.torus_layer_positions)[neurons_with_input_idx]

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
                save_prefix=self.save_prefix
            )

        self.rf_center_map = [
            (
                (x + (self.layer_size / 2.)) / float(self.layer_size) * self.input_stimulus.shape[1],
                (y + (self.layer_size / 2.)) / float(self.layer_size) * self.input_stimulus.shape[0]
            )
            for (x, y) in positions_with_input
        ]

        self.ff_weight_mat, self.input_recon = create_connections_rf(
            self.input_stimulus,
            neurons_with_input,
            self.rf_center_map,
            self.neuron_to_tuning_map,
            self.torus_inh_nodes,
            total_num_target=int(self.num_sensory),
            synaptic_strength=self.ff_weight,
            tuning_function=self.tuning_function,
            p_rf=self.p_rf,
            rf_size=self.rf_size,
            target_layer_size=self.layer_size,
            calc_error=self.use_input_neurons,
            use_dc=self.use_dc,
            plot_src_target=self.plot_rf_relation,
            retina_size=self.input_stimulus.shape,
            save_plot=self.save_plots,
            save_prefix=self.save_prefix,
            color_mask=self.color_map
        )

    def set_same_input_current(self):
        """
        If required, the same input can be set for all sensory neurons. Input is either DC or Poisson spike train,
        depending on the flag use_d
        :return: None
        """
        if self.verbosity > 0:
            print("\n#####################\tSet same input current to all sensory neurons")

        # Check ups
        if self.torus_layer is None:
            raise ValueError("The neural sheet has not been created yet. Run create_layer")
        if self.ff_weight is None:
            raise ValueError("The feedforward weight must not be None. Run determine_ffweight")
        if self.rf_size[0] < 0 or self.rf_size[1] < 0:
            raise ValueError("The size and shape of the receptive field must not be negative.")

        same_input_current(self.torus_layer, self.p_rf, self.ff_weight, rf_size=self.rf_size, use_dc=self.use_dc)

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
            print("\n#####################\tSimulate")

        # Check ups
        if self.spike_detect is None:
            raise ValueError("The spike detector must not be None. Run create_layer")
        nest.Simulate(float(simulation_time))
        # Get network response in spikes
        data_sp = nest.GetStatus(self.spike_detect, keys="events")[0]
        spikes_s = data_sp["senders"]
        time_s = data_sp["times"]

        if use_equilibrium:
            time_s = np.asarray(time_s)
            spikes_s = np.asarray(spikes_s)[time_s > eq_time].tolist()
            time_s = time_s[time_s > eq_time].tolist()

        firing_rates = get_firing_rates(spikes_s, self.torus_layer_nodes, simulation_time)
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

    def connect_distribution(self, plot_name="in_out_deg_dist.png"):
        """
        Plot the in-/outdegree distribution in the network
        :param plot_name: Name of the plot
        :return: None
        """
        # Check ups
        if self.torus_layer_nodes is None:
            raise ValueError("The sensory nodes have not been created yet. Run create_layer")

        in_degree, out_degree, _, _, _, _ = get_in_out_degree(self.torus_layer_nodes)

        in_deg_dist = OrderedDict(sorted(Counter(in_degree).items()))
        out_deg_dist = OrderedDict(sorted(Counter(out_degree).items()))

        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].bar(list(in_deg_dist.keys()), list(in_deg_dist.values()))
        ax[0].set_xlabel("Indegree total")
        ax[0].set_ylabel("Number of nodes")

        ax[1].bar(list(out_deg_dist.keys()), list(out_deg_dist.values()))
        ax[1].set_xlabel("Outdegree total")
        ax[1].set_ylabel("Number of nodes")

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
            print("\n#####################\tCreate adjacency matrix for sensory-to-sensory connections")
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
            print("\n#####################\tSet synaptic weights for sensory to sensory neurons")
        adj_sens_sens_mat = self.get_sensory_weight_mat()
        set_synaptic_strength(
            self.torus_layer_nodes,
            adj_sens_sens_mat,
            cap_s=weight,
            divide_by_num_connect=divide_by_num_connect
        )

    def set_input_stimulus(self, img):
        """
        Set new input stimulus and recomputes the injected current / Poisson spike rates
        :param img: The new input image
        :return: None
        """
        self.input_stimulus = img
        self.create_retina()

    # #################################################################################################################
    # Abstract methods
    # #################################################################################################################

    def create_network(self):
        """
        Creates the network and sets up all necessary connections
        :return: None
        """
        # Reset Nest Kernel
        nest.ResetKernel()
        self.create_layer()
        self.create_orientation_map()
        if not self.all_same_input_current:
            self.create_retina()
        else:
            self.set_same_input_current()


class RandomNetwork(NeuronalNetworkBase):
    def __init__(
            self,
            input_stimulus,
            p_random=0.005,
            num_sensory=int(1e4),
            layer_size=8.,
            verbosity=0,
            **kwargs
    ):
        """
        Random network class
        :param input_stimulus: The input stimulus
        :param p_random: Connection probability to connect to another neuron in the network
        :param num_sensory: Number of sensory nodes in the sheet
        :param layer_size: Size of the layer
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
            input_stimulus,
            num_sensory=num_sensory,
            layer_size=layer_size,
            spacing_perlin=spacing_perlin,
            resolution_perlin=resolution_perlin,
            verbosity=verbosity,
            **kwargs
        )

        self.p_random = p_random
        self.plot_random_connections = False if verbosity < 4 else True

    def create_random_connections(self):
        """
        Establish random connections to other nodes in the network
        :return: None
        """
        if self.verbosity > 0:
            print("\n#####################\tCreate random connections")

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

    def create_network(self):
        """
        Create the network and establish the connections. Calls create function of parent class
        :return: None
        """
        NeuronalNetworkBase.create_network(self)
        if self.use_input_neurons:
            return
        self.create_random_connections()


class LocalNetwork(NeuronalNetworkBase):
    ACCEPTED_LOC_CONN = ["circular", "sd"]

    def __init__(
            self,
            input_stimulus,
            p_loc=0.5,
            r_loc=0.5,
            loc_connection_type="circular",
            verbosity=0,
            **kwargs
    ):
        """
        Class that establishes local connections with locally clustered tuning specfic neurons
        :param input_stimulus: The input stimulus
        :param p_loc: Connection probability to connect to another neuron within the local radius
        :param r_loc: Radius within which a local connection is established
        :param loc_connection_type: Connection policy for local connections. This can be any value in the
        ACCEPTED_LOC_CONN list. Circ are circular connections, whereas sd are stimulus dependent connections
        :param verbosity: Determines the amount of output and created plots
        :param kwargs: Arguments that are passed to parent class
        """
        self.__dict__.update(kwargs)
        NeuronalNetworkBase.__init__(
            self,
            input_stimulus,
            verbosity=verbosity,
            **kwargs
        )

        self.p_loc = p_loc
        self.r_loc = r_loc
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
                print("\n#####################\tCreate local stimulus dependent connections")
            # Connection specific check up
            if self.neuron_to_tuning_map is None:
                raise ValueError("The mapping from neuron to orientation tuning must be created first."
                                 " Run create_orientation_map")
            if self.tuning_to_neuron_map is None:
                raise ValueError("The mapping from orientation tuning to neuron must be created first."
                                 " Run create_orientation_map")

            create_stimulus_based_local_connections(
                self.torus_layer,
                self.torus_layer_tree,
                self.neuron_to_tuning_map,
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
                print("\n#####################\tCreate local circular connections")
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
        # Check ups
        if self.torus_layer_nodes is None:
            raise ValueError("The sensory nodes have not been created yet. Run create_layer")

        in_degree, out_degree, in_degree_loc, out_degree_loc, _, _ = get_in_out_degree(
            self.torus_layer_nodes,
            node_tree=self.torus_layer_tree,
            node_pos=self.torus_layer_positions,
            r_loc=self.r_loc,
            size_layer=self.layer_size
        )

        in_deg_dist = OrderedDict(sorted(Counter(in_degree).items()))
        out_deg_dist = OrderedDict(sorted(Counter(out_degree).items()))

        in_deg_dist_loc = OrderedDict(sorted(Counter(in_degree_loc).items()))
        out_deg_dist_loc = OrderedDict(sorted(Counter(out_degree_loc).items()))

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0][0].bar(list(in_deg_dist.keys()), list(in_deg_dist.values()))
        ax[0][0].set_xlabel("Indegree total")
        ax[0][0].set_ylabel("Number of nodes")

        ax[0][1].bar(list(out_deg_dist.keys()), list(out_deg_dist.values()))
        ax[0][1].set_xlabel("Outdegree total")
        ax[0][1].set_ylabel("Number of nodes")

        ax[1][0].bar(list(in_deg_dist_loc.keys()), list(in_deg_dist_loc.values()))
        ax[1][0].set_xlabel("Indegree local")
        ax[1][0].set_ylabel("Number of nodes")

        ax[1][1].bar(list(out_deg_dist_loc.keys()), list(out_deg_dist_loc.values()))
        ax[1][1].set_xlabel("Outdegree local")
        ax[1][1].set_ylabel("Number of nodes")

        if self.save_plots:
            curr_dir = os.getcwd()
            Path(curr_dir + "/figures/in-out-dist/").mkdir(parents=True, exist_ok=True)
            plt.savefig(curr_dir + "/figures/in-out-dist/%s_%s" % (self.save_prefix, plot_name))
        else:
            plt.show()

    def create_network(self):
        """
        Creates the network and class the create function of the parent class
        :return: None
        """
        NeuronalNetworkBase.create_network(self)
        if self.use_input_neurons:
            return
        self.create_local_connections()


class PatchyNetwork(LocalNetwork):
    ACCEPTED_LR_CONN = ["random", "sd"]

    def __init__(
            self,
            input_stimulus,
            p_lr=0.2,
            num_patches=3,
            lr_connection_type="sd",
            verbosity=0,
            **kwargs
    ):
        """
        Class for patchy networks with long-range patchy connections
        :param input_stimulus: Input image
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
            input_stimulus,
            verbosity=verbosity,
            **kwargs
        )

        self.p_lr = p_lr
        self.num_patches = num_patches
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
                print("\n#####################\tCreate long-range patchy stimulus dependent connections")
            # Connection specific check up
            if self.torus_layer_tree is None:
                raise ValueError("The torus layer organised in a tree must be created first. Run create_layer")
            if self.neuron_to_tuning_map is None:
                raise ValueError("The mapping from neuron to orientation tuning must be created first."
                                 " Run create_orientation_map")
            if self.tuning_to_neuron_map is None:
                raise ValueError("The mapping from orientation tuning to neuron must be created first."
                                 " Run create_orientation_map")

            patchy_connect_dict = {"rule": "pairwise_bernoulli", "p": self.p_lr}
            create_stimulus_based_patches_random(
                self.torus_layer,
                self.neuron_to_tuning_map,
                self.tuning_to_neuron_map,
                self.torus_inh_nodes,
                self.torus_layer_tree,
                r_loc=self.r_loc,
                connect_dict=patchy_connect_dict,
                num_patches=self.num_patches,
                plot=self.plot_patchy_connections,
                save_plot=self.save_plots,
                save_prefix=self.save_prefix,
                color_mask=self.color_map
            )

        if self.lr_connection_type == "random":
            if self.verbosity > 0:
                print("\n#####################\tCreate long-range patchy random connections")
            create_random_patches(
                self.torus_layer,
                self.torus_inh_nodes,
                r_loc=self.r_loc,
                p_loc=self.p_loc,
                cap_s=self.cap_s,
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
        # Check ups
        if self.torus_layer_nodes is None:
            raise ValueError("The sensory nodes have not been created yet. Run create_layer")

        in_degree, out_degree, in_degree_loc, out_degree_loc, in_degree_lr, out_degree_lr = get_in_out_degree(
            self.torus_layer_nodes,
            node_tree=self.torus_layer_tree,
            node_pos=self.torus_layer_positions,
            r_loc=self.r_loc,
            size_layer=self.layer_size
        )

        in_deg_dist = OrderedDict(sorted(Counter(in_degree).items()))
        out_deg_dist = OrderedDict(sorted(Counter(out_degree).items()))

        in_deg_dist_loc = OrderedDict(sorted(Counter(in_degree_loc).items()))
        out_deg_dist_loc = OrderedDict(sorted(Counter(out_degree_loc).items()))

        in_deg_dist_lr = OrderedDict(sorted(Counter(in_degree_lr).items()))
        out_deg_dist_lr = OrderedDict(sorted(Counter(out_degree_lr).items()))

        fig, ax = plt.subplots(3, 2, figsize=(10, 10))
        ax[0][0].bar(list(in_deg_dist.keys()), list(in_deg_dist.values()))
        ax[0][0].set_xlabel("Indegree total")
        ax[0][0].set_ylabel("Number of nodes")

        ax[0][1].bar(list(out_deg_dist.keys()), list(out_deg_dist.values()))
        ax[0][1].set_xlabel("Outdegree total")
        ax[0][1].set_ylabel("Number of nodes")

        ax[1][0].bar(list(in_deg_dist_loc.keys()), list(in_deg_dist_loc.values()))
        ax[1][0].set_xlabel("Indegree local")
        ax[1][0].set_ylabel("Number of nodes")

        ax[1][1].bar(list(out_deg_dist_loc.keys()), list(out_deg_dist_loc.values()))
        ax[1][1].set_xlabel("Outdegree local")
        ax[1][1].set_ylabel("Number of nodes")

        ax[2][0].bar(list(in_deg_dist_lr.keys()), list(in_deg_dist_lr.values()))
        ax[2][0].set_xlabel("Indegree long-range")
        ax[2][0].set_ylabel("Number of nodes")

        ax[2][1].bar(list(out_deg_dist_lr.keys()), list(out_deg_dist_lr.values()))
        ax[2][1].set_xlabel("Outdegree long-range")
        ax[2][1].set_ylabel("Number of nodes")

        if self.save_plots:
            curr_dir = os.getcwd()
            Path(curr_dir + "/figures/in-out-dist/").mkdir(parents=True, exist_ok=True)
            plt.savefig(curr_dir + "/figures/in-out-dist/%s_%s" % (self.save_prefix, plot_name))
        else:
            plt.show()

    def create_network(self):
        """
        Create the network, establishes the connections and calls the create function of the parent class
        :return:
        """
        LocalNetwork.create_network(self)
        if self.use_input_neurons:
            return
        self.create_lr_connections()


def network_factory(input_stimulus, network_type=NETWORK_TYPE["local_circ_patchy_sd"], **kwargs):
    """
    Factory function to instantiate an neuronal network object
    :param input_stimulus: The iput image
    :param network_type: The network type. The value can be any integer defined in the NETWORK_TYPE dictionary
    :param kwargs: The parameters passed to the network
    :return: The network object
    """
    if network_type == NETWORK_TYPE["random"]:
        network = RandomNetwork(
            input_stimulus,
            **kwargs
        )
    elif network_type == NETWORK_TYPE["local_circ"]:
        network = LocalNetwork(
            input_stimulus,
            loc_connection_type="circular",
            **kwargs
        )
    elif network_type == NETWORK_TYPE["local_sd"]:
        network = LocalNetwork(
            input_stimulus,
            loc_connection_type="sd",
            **kwargs
        )
    elif network_type == NETWORK_TYPE["local_circ_patchy_random"]:
        network = PatchyNetwork(
            input_stimulus,
            loc_connection_type="circular",
            lr_connection_type="random",
            **kwargs
        )
    elif network_type == NETWORK_TYPE["local_circ_patchy_sd"]:
        network = PatchyNetwork(
            input_stimulus,
            loc_connection_type="circular",
            lr_connection_type="sd",
            **kwargs
        )
    elif network_type == NETWORK_TYPE["local_sd_patchy_sd"]:
        network = PatchyNetwork(
            input_stimulus,
            loc_connection_type="sd",
            lr_connection_type="sd",
            **kwargs
        )
    elif network_type == NETWORK_TYPE["input_only"]:
        network = NeuronalNetworkBase(
            input_stimulus,
            **kwargs
        )
    else:
        raise ValueError("Network type %s is not accpepted" % network_type)

    return network

