#!/usr/bin/python3
# -*- coding: utf-8 -*-

from modules.networkConstruction import *
from modules.createStimulus import *
from modules.networkAnalysis import *

from scipy.spatial import KDTree
import nest

nest.set_verbosity("M_ERROR")

NETWORK_TYPE = {
    "random": 0,
    "local_circ": 1,
    "local_sd": 2,
    "local_circ_patchy_sd": 3,
    "local_circ_patchy_random": 4,
    "local_sd_patchy_sd": 5
}


class NeuronalNetworkBase:
    def __init__(
            self,
            input_stimulus,
            num_sensory=int(1e4),
            ratio_inh_neurons=5,
            num_stim_discr=4,
            cap_s=1.,
            inh_weight=-15.,
            p_rf=0.3,
            rf_size=None,
            use_continuous_tuning=True,
            all_same_input_current=False,
            pot_threshold=-55.,
            pot_reset=-70.,
            capacitance=80.,
            time_constant=20.,
            layer_size=8.,
            spacing_perlin=0.01,
            resolution_perlin=(15, 15),
            verbosity=0,
            save_plots=False,
            **kwargs
    ):
        self.input_stimulus = input_stimulus
        self.num_sensory = int(num_sensory)
        self.ratio_inh_neurons = ratio_inh_neurons
        self.num_stim_discr = num_stim_discr
        self.cap_s = cap_s
        self.inh_weight = inh_weight
        self.p_rf = p_rf
        self.rf_size = rf_size

        if self.rf_size is None:
            self.rf_size = (input_stimulus.shape[0] // 4, input_stimulus.shape[1] // 4)

        self.all_same_input_current = all_same_input_current
        self.use_continuous_tuning = use_continuous_tuning

        self.pot_threshold = pot_threshold
        self.pot_reset = pot_reset
        self.capacitance = capacitance
        self.time_constant = time_constant
        self.layer_size = layer_size

        self.spacing_perlin = spacing_perlin
        self.resolution_perlin = resolution_perlin

        self.verbosity = verbosity
        self.save_plots = save_plots

        self.plot_rf_relation = False if verbosity < 4 else True
        self.plot_tuning_map = False if verbosity < 4 else True

        self.ff_weight = None
        self.determine_ffweight()

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

    def determine_ffweight(self):
        if self.verbosity > 0:
            print("\n#####################\tDetermine feedforward weight")

        self.ff_weight = determine_ffweight(self.rf_size)

    # #################################################################################################################
    # Network creation
    # #################################################################################################################

    def create_layer(self):
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
            size=self.num_sensory // self.ratio_inh_neurons
        ).tolist()

    def create_orientation_map(self):
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
        if self.resolution_perlin[0] < self.num_stim_discr or self.resolution_perlin[1] < self.num_stim_discr:
            raise ValueError("The resolution of the mesh for the Perlin noise is not set to a meaningful value."
                             " Set it larger than %s. Current value is %s"
                             % (self.num_stim_discr, self.resolution_perlin))

        (self.tuning_to_neuron_map,
         self.neuron_to_tuning_map,
         self.color_map) = create_perlin_stimulus_map(
            self.torus_layer,
            self.torus_inh_nodes,
            num_stimulus_discr=self.num_stim_discr,
            plot=self.plot_tuning_map,
            spacing=self.spacing_perlin,
            resolution=self.resolution_perlin,
            save_plot=self.save_plots
        )

    def create_retina(self):
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

        if self.rf_center_map is None:
            self.rf_center_map = [
                (
                    (x + (self.layer_size / 2.)) / float(self.layer_size) * self.input_stimulus.shape[1],
                    (y + (self.layer_size / 2.)) / float(self.layer_size) * self.input_stimulus.shape[0]
                )
                for (x, y) in self.torus_layer_positions
            ]

        # Create connections to receptive field
        if self.verbosity > 0:
            print("\n#####################\tCreate connections between receptors and sensory neurons")

        self.ff_weight_mat = create_connections_rf(
            self.input_stimulus,
            self.torus_layer,
            self.rf_center_map,
            self.neuron_to_tuning_map,
            self.torus_inh_nodes,
            synaptic_strength=self.ff_weight,
            use_continuous_tuning=self.use_continuous_tuning,
            p_rf=self.p_rf,
            rf_size=self.rf_size,
            plot_src_target=self.plot_rf_relation,
            retina_size=self.input_stimulus.shape,
            save_plot=self.save_plots
        )

    def set_same_input_current(self):
        if self.verbosity > 0:
            print("\n#####################\tSet same input current to all sensory neurons")

        # Check ups
        if self.torus_layer is None:
            raise ValueError("The neural sheet has not been created yet. Run create_layer")
        if self.ff_weight is None:
            raise ValueError("The feedforward weight must not be None. Run determine_ffweight")
        if self.rf_size[0] < 0 or self.rf_size[1] < 0:
            raise ValueError("The size and shape of the receptive field must not be negative.")

        same_input_current(self.torus_layer, self.p_rf, self.ff_weight, rf_size=self.rf_size)

    # #################################################################################################################
    # Simulate
    # #################################################################################################################

    def simulate(self, simulation_time=250.):
        if self.verbosity > 0:
            print("\n#####################\tSimulate")

        # Check ups
        if self.spike_detect is None:
            raise ValueError("The spike detector must not be None. Run create_layer")
        nest.Simulate(simulation_time)
        # Get network response in spikes
        data_sp = nest.GetStatus(self.spike_detect, keys="events")[0]
        spikes_s = data_sp["senders"]
        time_s = data_sp["times"]

        firing_rates = get_firing_rates(spikes_s, self.torus_layer_nodes, simulation_time)
        return firing_rates, (spikes_s, time_s)

    # #################################################################################################################
    # Plot connections
    # #################################################################################################################

    def plot_connections_node(self, node_idx=0, plot_name="all_connections.png"):
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
            color_mask=self.color_map
        )
    # #################################################################################################################
    # Getter / Setter
    # #################################################################################################################

    def get_sensory_weight_mat(self):
        if self.verbosity > 0:
            print("\n#####################\tCreate adjacency matrix for sensory-to-sensory connections")
        if self.adj_sens_sens_mat is None:
            self.adj_sens_sens_mat = create_adjacency_matrix(self.torus_layer_nodes, self.torus_layer_nodes)
        return self.adj_sens_sens_mat

    def set_recurrent_weight(self, weight, divide_by_num_connect=False):
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
        self.input_stimulus = img
        self.create_retina()

    # #################################################################################################################
    # Abstract methods
    # #################################################################################################################

    def create_network(self):
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
            p_random=0.001,
            num_sensory=int(1e4),
            ratio_inh_neurons=5,
            num_stim_discr=4,
            cap_s=1.,
            inh_weight=-15.,
            p_rf=0.3,
            rf_size=None,
            use_continuous_tuning=True,
            pot_threshold=-55.,
            pot_reset=-70.,
            capacitance=80.,
            layer_size=8.,
            verbosity=0,
            save_plots=False,
            **kwargs
    ):
        spacing_perlin = layer_size / np.sqrt(num_sensory)
        res_perlin = int(layer_size * np.sqrt(num_sensory))
        resolution_perlin = (res_perlin, res_perlin)
        self.__dict__.update(kwargs)
        NeuronalNetworkBase.__init__(
            self,
            input_stimulus,
            num_sensory=num_sensory,
            ratio_inh_neurons=ratio_inh_neurons,
            num_stim_discr=num_stim_discr,
            cap_s=cap_s,
            inh_weight=inh_weight,
            p_rf=p_rf,
            rf_size=rf_size,
            use_continuous_tuning=use_continuous_tuning,
            pot_threshold=pot_threshold,
            pot_reset=pot_reset,
            capacitance=capacitance,
            layer_size=layer_size,
            spacing_perlin=spacing_perlin,
            resolution_perlin=resolution_perlin,
            verbosity=verbosity,
            save_plots=save_plots,
            **kwargs
        )

        self.p_random = p_random
        self.plot_random_connections = False if verbosity < 4 else True

    def create_random_connections(self):
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
            color_mask=self.color_map
        )

    def create_network(self):
        NeuronalNetworkBase.create_network(self)
        self.create_random_connections()


class LocalNetwork(NeuronalNetworkBase):
    ACCEPTED_LOC_CONN = ["circular", "sd"]

    def __init__(
            self,
            input_stimulus,
            p_loc=0.5,
            r_loc=0.5,
            loc_connection_type="circular",
            num_sensory=int(1e4),
            ratio_inh_neurons=5,
            num_stim_discr=4,
            cap_s=1.,
            inh_weight=-15.,
            p_rf=0.3,
            rf_size=None,
            use_continuous_tuning=True,
            pot_threshold=-55.,
            pot_reset=-70.,
            capacitance=80.,
            layer_size=8.,
            spacing_perlin=0.01,
            resolution_perlin=(15, 15),
            verbosity=0,
            save_plots=False,
            **kwargs
    ):
        self.__dict__.update(kwargs)
        NeuronalNetworkBase.__init__(
            self,
            input_stimulus,
            num_sensory=num_sensory,
            ratio_inh_neurons=ratio_inh_neurons,
            num_stim_discr=num_stim_discr,
            cap_s=cap_s,
            inh_weight=inh_weight,
            p_rf=p_rf,
            rf_size=rf_size,
            use_continuous_tuning=use_continuous_tuning,
            pot_threshold=pot_threshold,
            pot_reset=pot_reset,
            capacitance=capacitance,
            layer_size=layer_size,
            spacing_perlin=spacing_perlin,
            resolution_perlin=resolution_perlin,
            verbosity=verbosity,
            save_plots=save_plots,
            **kwargs
        )

        self.p_loc = p_loc
        self.r_loc = r_loc
        self.loc_connection_type = loc_connection_type.lower()
        if self.loc_connection_type not in LocalNetwork.ACCEPTED_LOC_CONN:
            raise ValueError("The passed connection type %s is not accepted." % self.loc_connection_type)

        self.plot_local_connections = False if verbosity < 4 else True

    def create_local_connections(self):
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
                save_plot=self.save_plots
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
                save_plot=self.save_plots
            )
        else:
            raise ValueError("The passed connection type %s is not accepted." % self.loc_connection_type)

    def create_network(self):
        NeuronalNetworkBase.create_network(self)
        self.create_local_connections()


class PatchyNetwork(LocalNetwork):
    ACCEPTED_LR_CONN = ["random", "sd"]

    def __init__(
            self,
            input_stimulus,
            p_lr=0.2,
            num_patches=3,
            lr_connection_type="sd",
            p_loc=0.5,
            r_loc=0.5,
            loc_connection_type="circular",
            num_sensory=int(1e4),
            ratio_inh_neurons=5,
            num_stim_discr=4,
            cap_s=1.,
            inh_weight=-15.,
            p_rf=0.3,
            rf_size=None,
            use_continuous_tuning=True,
            pot_threshold=-55.,
            pot_reset=-70.,
            capacitance=80.,
            layer_size=8.,
            spacing_perlin=0.01,
            resolution_perlin=(15, 15),
            verbosity=0,
            save_plots=False,
            **kwargs
    ):
        self.__dict__.update(kwargs)
        LocalNetwork.__init__(
            self,
            input_stimulus,
            p_loc=p_loc,
            r_loc=r_loc,
            loc_connection_type=loc_connection_type,
            num_sensory=num_sensory,
            ratio_inh_neurons=ratio_inh_neurons,
            num_stim_discr=num_stim_discr,
            cap_s=cap_s,
            inh_weight=inh_weight,
            p_rf=p_rf,
            rf_size=rf_size,
            use_continuous_tuning=use_continuous_tuning,
            pot_threshold=pot_threshold,
            pot_reset=pot_reset,
            capacitance=capacitance,
            layer_size=layer_size,
            spacing_perlin=spacing_perlin,
            resolution_perlin=resolution_perlin,
            verbosity=verbosity,
            save_plots=save_plots,
            **kwargs
        )

        self.p_lr = p_lr
        self.num_patches = num_patches
        self.lr_connection_type = lr_connection_type.lower()
        if self.lr_connection_type not in PatchyNetwork.ACCEPTED_LR_CONN:
            raise ValueError("%s is not an accepted long-range connection type" % self.lr_connection_type)

        self.plot_patchy_connections = False if verbosity < 4 else True

    def create_lr_connections(self):
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
                color_mask=self.color_map
            )

    def create_network(self):
        LocalNetwork.create_network(self)
        self.create_lr_connections()


def network_factory(input_stimulus, network_type=NETWORK_TYPE["local_circ_patchy_sd"], **kwargs):
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
    else:
        raise ValueError("Network type %s is not accpepted" % network_type)

    return network

