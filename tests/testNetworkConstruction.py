#!/usr/bin/python3
import unittest

import modules.networkConstruction as nc
import modules.createStimulus as cs
import numpy as np

import nest.topology as tp
import nest
nest.set_verbosity("M_ERROR")

np.random.seed(0)


class NetworkConstructionTest(unittest.TestCase):
    def setUp(self):
        self.num_neurons = 1500
        self.neuron_type = "iaf_psc_delta"
        self.rest_pot = -1.
        self.threshold_pot = 1e2
        self.time_const = 22.
        self.capacitance = 1e4

        self.num_stimulus_discr = 4
        self.size_layer = 2.

        self.input_stimulus = cs.image_with_spatial_correlation(size_img=(20, 20), radius=3, num_circles=80)
        organise_on_grid = True

        self.torus_layer, self.spike_det, self.multi = nc.create_torus_layer_uniform(
            num_neurons=self.num_neurons,
            neuron_type=self.neuron_type,
            rest_pot=self.rest_pot,
            threshold_pot=self.threshold_pot,
            time_const=self.time_const,
            capacitance=self.capacitance,
            size_layer=self.size_layer
        )
        self.torus_nodes = nest.GetNodes(self.torus_layer)[0]
        self.min_id_torus = min(self.torus_nodes)

        (self.tuning_to_neuron_map,
         self.neuron_to_tuning_map,
         self.tuning_weight_vector,
         _) = nc.create_stimulus_tuning_map(
            self.torus_layer,
            num_stimulus_discr=self.num_stimulus_discr
        )

        self.retina = nc.create_input_current_generator(
            self.input_stimulus,
            organise_on_grid=organise_on_grid
        )

        self.receptors = nest.GetNodes(self.retina)[0]

    # ################################################################################################################
    # Functions used for thesis simulations
    # ################################################################################################################

    def test_create_distinct_sublayer_boxes(self):
        size_boxes = 4.
        size_layer = 8.
        expected_anchors = np.asarray([[-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0], [2.0, 2.0]])
        expected_box_mask = {"lower_left": [-4. / 2., -4. / 2.],
                     "upper_right": [4. / 2., 4. / 2.]}
        anchors, box_mask = nc.create_distinct_sublayer_boxes(size_boxes, size_layer=size_layer)

        self.assertTrue(
            np.all(anchors == np.asarray(expected_anchors)),
            "Anchors don't mach the expected anchors. Actual anchors %s,"
            " expected anchors %s" % (anchors, expected_anchors)
        )
        self.assertEqual(
            box_mask,
            expected_box_mask,
            "Box masks don't match.  Actual mask %s, expected mask %s" % (box_mask, expected_anchors)
        )

    def test_create_torus_layer_uniform(self):
        # Create simple layer
        num_neurons = 3200
        neuron_type = "iaf_psc_delta"
        rest_pot = -1.
        threshold_pot = 1e2
        time_const = 22.
        capacitance = 1e4

        layer, spike_det, multi = nc.create_torus_layer_uniform(
            num_neurons=num_neurons,
            neuron_type=neuron_type,
            rest_pot=rest_pot,
            threshold_pot=threshold_pot,
            time_const=time_const,
            capacitance=capacitance
        )

        # Check whether right amount of neurons was created
        nodes = nest.GetNodes(layer)[0]
        self.assertEqual(
            len(nodes),
            num_neurons,
            "Wrong number of neurons created. Actual number %s, expected number %s" % (len(nodes), num_neurons)
        )

        # Check values of the neurons
        nodes_model = nest.GetStatus(nodes, "model")
        self.assertTrue(
            np.all(np.asarray(nodes_model) == neuron_type),
            "Wrong neuron type chosen. Actual neuron type %s, expected neuron type %s" % (nodes_model[0], neuron_type)
        )

        nodes_v_m = nest.GetStatus(nodes, "V_m")
        self.assertTrue(
            np.all(np.asarray(nodes_v_m) == rest_pot),
            "Membrane potential is set to the wrong value. Actual min value %s,"
            " actual max value %s, expected values %s" %
            (np.asarray(nodes_v_m).min(), np.asarray(nodes_v_m).max, rest_pot)
        )

        nodes_e_l = nest.GetStatus(nodes, "E_L")
        self.assertTrue(
            np.all(np.asarray(nodes_e_l) == rest_pot),
            "Resting potential is set to the wrong value. Actual min value %s,"
            " actual max values %s, expected value %s" %
            (np.asarray(nodes_e_l).min(), np.asarray(nodes_e_l).max(), rest_pot)
        )

        nodes_c_m = nest.GetStatus(nodes, "C_m")
        self.assertTrue(
            np.all(np.asarray(nodes_c_m) == capacitance),
            "Capacitance is set to the wrong value. Actual min value %s,"
            " actual max values %s, expected value %s" %
            (np.asarray(nodes_c_m).min(), np.asarray(nodes_c_m).max(), capacitance)
        )

        nodes_tau = nest.GetStatus(nodes, "tau_m")
        self.assertTrue(
            np.all(np.asarray(nodes_tau) == time_const),
            "Time constant is set to the wrong value. Actual min value %s,"
            " actual max values %s, expected value %s" %
            (np.asarray(nodes_tau).min(), np.asarray(nodes_tau).max(), time_const)
        )

        nodes_v_th = nest.GetStatus(nodes, "V_th")
        self.assertTrue(
            np.all(np.asarray(nodes_v_th) == threshold_pot),
            "Threshold potential is set to the wrong value. Actual min value %s,"
            " actual max values %s, expected value %s" %
            (np.asarray(nodes_v_th).min(), np.asarray(nodes_v_th).max(), threshold_pot)
        )

        nodes_v_reset = nest.GetStatus(nodes, "V_reset")
        self.assertTrue(
            np.all(np.asarray(nodes_v_reset) == rest_pot),
            "Reset potential is set to the wrong value. Actual min value %s,"
            " actual max values %s, expected value %s" %
            (np.asarray(nodes_v_reset).min(), np.asarray(nodes_v_reset).max(), rest_pot)
        )

        # Check connections of the spike detector and multimeter
        spike_connect = nest.GetConnections(source=nodes, target=spike_det)
        self.assertNotEqual(len(spike_connect), 0, "Connection between neurons and spike detector doesn't exist")

        no_spike_connect = nest.GetConnections(source=spike_det, target=nodes)
        self.assertEqual(len(no_spike_connect), 0, "Connection between spike detector and neurons does exist")

        no_multi_connect = nest.GetConnections(source=nodes, target=multi)
        self.assertEqual(len(no_multi_connect), 0, "Connection between neurons and multimeter does exist")

        multi_connect = nest.GetConnections(source=multi, target=nodes)
        self.assertNotEqual(len(multi_connect), 0, "Connection between multimeter and neurons doesn't exist")

    def test_create_local_circular_connections(self):
        r_loc = 0.3
        p_loc = 0.4
        nc.create_local_circular_connections(self.torus_layer, r_loc=r_loc, p_loc=p_loc)
        connect = nest.GetConnections(self.torus_layer)
        for c in connect:
            s = nest.GetStatus(c, "source")
            t = nest.GetStatus(c, "target")
            self.assertLessEqual(tp.Distance(s, t), r_loc)
            nest.Disconnect(s, t)

    def test_create_stimulus_tuning_map(self):
        num_stimulus_discr = 4
        tuning_to_neuron_map, neuron_to_tuning_map, tuning_weight_vector, _ = nc.create_stimulus_tuning_map(
            self.torus_layer,
            num_stimulus_discr=num_stimulus_discr,
        )

        for tuning, area in tuning_to_neuron_map.items():
            for neurons in area:
                for n in neurons:
                    self.assertEqual(
                        neuron_to_tuning_map[n],
                        tuning,
                        "Neuron tuning mismatch. Tuning to neuron map has tuning %s,"
                        " but neuron to tuning map has %s" %
                        (tuning, neuron_to_tuning_map[n])
                    )
                    self.assertEqual(
                        tuning / float(num_stimulus_discr - 1),
                        tuning_weight_vector[n - self.min_id_torus],
                        "Neuron weight mismatch. Actual tuning weight %s,"
                        " but expected tuning weight is %s" %
                        (tuning_weight_vector[n - self.min_id_torus], tuning / float(num_stimulus_discr - 1))
                    )

    def test_create_stimulus_based_local_connections(self):
        r_loc = 0.2
        connect_dict = {"rule": "pairwise_bernoulli", "p": 0.8}
        nc.create_stimulus_based_local_connections(
            self.torus_layer,
            self.neuron_to_tuning_map,
            self.tuning_to_neuron_map,
            r_loc=r_loc,
            connect_dict=connect_dict
        )

        connect = nest.GetConnections(self.torus_layer)
        for c in connect:
            s = nest.GetStatus(c, "source")
            t = nest.GetStatus(c, "target")
            self.assertLessEqual(tp.Distance(s, t), r_loc)
            self.assertEqual(self.neuron_to_tuning_map[s], self.neuron_to_tuning_map[t])
            nest.Disconnect(s, t)

    def test_create_stimulus_based_patches_random(self):
        num_patches = 2
        r_loc = 0.5
        p_loc = 0.7
        connect_dict = None
        p_p = None

        # Without connection dict
        nc.create_stimulus_based_patches_random(
            self.torus_layer,
            self.neuron_to_tuning_map,
            self.tuning_to_neuron_map,
            num_patches=num_patches,
            r_loc=r_loc,
            p_loc=p_loc,
            p_p=p_p,
            connect_dict=connect_dict,
        )

        connect = nest.GetConnections(self.torus_layer)
        for c in connect:
            s = nest.GetStatus(c, "source")
            t = nest.GetStatus(c, "target")
            self.assertLessEqual(tp.Distance(s, t), self.size_layer)
            self.assertGreaterEqual(tp.Distance(s, t), r_loc)
            self.assertEqual(self.neuron_to_tuning_map[s], self.neuron_to_tuning_map[t])
            nest.Disconnect(s, t)

        # With connection dict
        connect_dict = {"rule": "pairwise_bernoulli", "p": 0.7}

        nc.create_stimulus_based_patches_random(
            self.torus_layer,
            self.neuron_to_tuning_map,
            self.tuning_to_neuron_map,
            num_patches=num_patches,
            r_loc=r_loc,
            p_loc=p_loc,
            p_p=p_p,
            connect_dict=connect_dict,
        )

        connect = nest.GetConnections(self.torus_layer)
        for c in connect:
            s = nest.GetStatus(c, "source")
            t = nest.GetStatus(c, "target")
            self.assertLessEqual(tp.Distance(s, t), self.size_layer)
            self.assertGreaterEqual(tp.Distance(s, t), r_loc)
            self.assertEqual(self.neuron_to_tuning_map[s], self.neuron_to_tuning_map[t])
            nest.Disconnect(s, t)

    def test_create_input_current_generator(self):
        input_stimulus = cs.image_with_spatial_correlation(size_img=(50, 50), radius=3, num_circles=80)
        organise_on_grid = True
        retina = nc.create_input_current_generator(
            input_stimulus,
            organise_on_grid=organise_on_grid
        )

        receptors = nest.GetNodes(retina)[0]
        amplitude = nest.GetStatus(receptors, "amplitude")
        self.assertTrue(
            np.all(np.asarray(amplitude) == input_stimulus.reshape(-1) * 1e12),
            "The input stimulus and the current in the retina don't match"
        )

    def test_create_connections_rf(self):
        rf_size = (10, 10)
        connect_dict = {"rule": "pairwise_bernoulli", "p": 1.}
        synaptic_strength = 1.

        positions = tp.GetPosition(self.torus_nodes)
        rf_centers = [
            (
                (x / (self.size_layer / 2.)) * self.input_stimulus.shape[1] / 2.,
                (y / (self.size_layer / 2.)) * self.input_stimulus.shape[0] / 2.
            )
            for (x, y) in positions
        ]
        rf_mask = {"lower_left": [-rf_size[0]/2., -rf_size[1]/2.], "upper_right": [rf_size[0]/2., rf_size[1]/2.]}

        adj_mat = nc.create_connections_rf(
            self.retina,
            self.torus_layer,
            rf_centers,
            self.neuron_to_tuning_map,
            rf_size=rf_size,
            connect_dict=connect_dict,
            synaptic_strength=synaptic_strength
        )

        # Check for lower equal since there can be less neurons at the edges of the retina
        self.assertLessEqual(
            np.max(adj_mat.sum(axis=0)),
            rf_size[0] * rf_size[1],
            "There are receptors to which no connections are established although they are in the receptive field"
        )

        tuning_disc_step = 255 / float(self.num_stimulus_discr)
        for n in self.torus_nodes:
            n_position = tp.GetPosition([n])[0]
            rf_center = ((n_position[0] / (self.size_layer / 2.)) * self.input_stimulus.shape[1] / 2.,
                         (n_position[1] / (self.size_layer / 2.)) * self.input_stimulus.shape[0] / 2.)
            rf = tp.SelectNodesByMask(self.retina, rf_center, tp.CreateMask("rectangular", specs=rf_mask))

            connect = nest.GetConnections(source=self.receptors, target=[n])
            source = nest.GetStatus(connect, "source")

            for s, c in zip(source, connect):
                self.assertIn(s, rf, "Receptor not in the receptive field")
                pos = tp.GetPosition([s])[0]
                grid_pos_x = int(self.input_stimulus.shape[0] / 2. + np.floor(pos[0]))
                grid_pos_y = int(self.input_stimulus.shape[1] / 2. - np.ceil(pos[1]))
                self.assertEqual(
                    tp.GetElement(self.retina, (grid_pos_x, grid_pos_y))[0],
                    s,
                    "Did not compute the correct grid position"
                )

                stimulus = self.input_stimulus[grid_pos_x, grid_pos_y]
                weight = nest.GetStatus([c], "weight")[0]
                stimulus_tuning = self.neuron_to_tuning_map[n]
                reacts_on_stimulus = stimulus_tuning * tuning_disc_step <= stimulus <\
                                     (stimulus_tuning + 1) * tuning_disc_step

                if reacts_on_stimulus:
                    self.assertEqual(
                        synaptic_strength,
                        weight,
                        "Synaptic weight is not set to the correct value. Actual value %s, expected value %s"
                        % (weight, synaptic_strength)
                    )
                else:
                    self.assertEqual(
                        weight,
                        0,
                        "Synaptic weight is not set to zero although not correct stimulus. Actual value %s" % weight
                    )
                nest.Disconnect([s], [n])

    def test_create_torus_layer_with_jitter(self):
        # Must be number that has a square root in N
        num_neurons = 144
        jitter = 0.01
        neuron_type = "iaf_psc_delta"

        layer = nc.create_torus_layer_with_jitter(
            num_neurons=num_neurons,
            jitter=jitter,
            neuron_type=neuron_type
        )

        nodes = nest.GetNodes(layer)[0]
        self.assertEqual(len(nodes), num_neurons, "Not the right number of nodes")

        model = nest.GetStatus(nodes, "model")
        for m in model:
            self.assertEqual(m, neuron_type, "Wrong neuron type set")

        mod_size = nc.R_MAX - jitter * 2
        step_size = mod_size / float(np.sqrt(num_neurons))
        coordinate_scale = np.arange(-mod_size / 2., mod_size / 2., step_size)
        grid = [[x, y] for y in coordinate_scale for x in coordinate_scale]

        for n in nodes:
            pos = np.asarray(tp.GetPosition([n])[0])
            sorted(grid, key=lambda l: np.linalg.norm(np.asarray(l) - pos))
            self.assertLessEqual(pos[0], np.abs(grid[0][0] + jitter), "Distance in x to original position too high")
            self.assertLessEqual(pos[1], np.abs(grid[0][1] + jitter), "Distance in y to original position too high")

    def test_create_distant_np_connections(self):
        r_loc = .2
        p_loc = .5

        nc.create_distant_np_connections(self.torus_layer, p_loc=p_loc, r_loc=r_loc)

        conn = nest.GetConnections(source=self.torus_nodes, target=self.torus_nodes)
        for c in conn:
            s = nest.GetStatus([c], "source")[0]
            t = nest.GetStatus([c], "target")[0]
            d = tp.Distance([s], [t])[0]
            self.assertLessEqual(d, self.size_layer / 2., "Distance is too large")
            self.assertGreaterEqual(d, r_loc, "Distance is too low")


if __name__ == '__main__':
    unittest.main()
