#!/usr/bin/python3
import unittest

import modules.networkConstruction as nc
import modules.createStimulus as cs
import modules.thesisUtils as tu
import modules.stimulusReconstruction as sr
import modules.networkAnalysis as na
import numpy as np
from scipy.spatial import KDTree

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
        self.organise_on_grid = True

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
        self.inh_nodes = np.random.choice(np.asarray(self.torus_nodes), self.num_neurons // 5, replace=False)
        self.min_id_torus = min(self.torus_nodes)
        self.torus_positions = tp.GetPosition(self.torus_nodes)
        self.torus_tree = KDTree(self.torus_positions)

        self.perlin_resolution = (15, 15)
        self.perlin_spacing = 0.1

        (self.tuning_to_neuron_map,
         self.neuron_to_tuning_map,
         self.tuning_weight_vector) = nc.create_perlin_stimulus_map(
            self.torus_layer,
            self.inh_nodes,
            num_stimulus_discr=self.num_stimulus_discr,
            resolution=self.perlin_resolution,
            spacing=self.perlin_spacing,
            plot=False,
            save_plot=False
        )

        self.retina = nc.create_input_current_generator(
            self.input_stimulus,
            organise_on_grid=self.organise_on_grid
        )

        self.receptors = nest.GetNodes(self.retina)[0]

    def reset(self):
        nest.ResetKernel()
        self.setUp()

    # ################################################################################################################
    # Functions used for thesis simulations
    # ################################################################################################################

    def test_create_distinct_sublayer_boxes(self):
        self.reset()

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
        self.reset()

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
        self.reset()

        r_loc = 0.3
        inh_weight = -5.
        cap_s = 2.
        connect_dict = {
            "rule": "pairwise_bernoulli",
            "p": 0.4,
        }

        nc.create_local_circular_connections(
            self.torus_layer,
            self.torus_tree,
            self.inh_nodes,
            inh_weight=inh_weight,
            cap_s=cap_s,
            connect_dict=connect_dict,
            r_loc=r_loc,
        )

        connect = nest.GetConnections(self.torus_nodes)
        for c in connect:
            s = nest.GetStatus([c], "source")[0]
            t = nest.GetStatus([c], "target")[0]
            if t in self.torus_nodes:
                d = tp.Distance([s], [t])[0]
                w = nest.GetStatus([c], "weight")[0]
                self.assertLessEqual(d, r_loc)
                if s in self.inh_nodes:
                    self.assertEqual(w, inh_weight, "Inhibitory weight was not set properly")
                else:
                    self.assertEqual(w, cap_s, "Excitatory weight was not set properly")

    def test_create_stimulus_tuning_map(self):
        self.reset()

        num_stimulus_discr = 4
        tuning_to_neuron_map, neuron_to_tuning_map, tuning_weight_vector, _ = nc.create_stimulus_tuning_map(
            self.torus_layer,
            num_stimulus_discr=num_stimulus_discr,
        )

        for tuning, area in tuning_to_neuron_map.items():
            for neuron in area:
                self.assertEqual(
                    neuron_to_tuning_map[neuron],
                    tuning,
                    "Neuron tuning mismatch. Tuning to neuron map has tuning %s,"
                    " but neuron to tuning map has %s" %
                    (tuning, neuron_to_tuning_map[neuron])
                )
                self.assertEqual(
                    tuning / float(num_stimulus_discr - 1),
                    tuning_weight_vector[neuron - self.min_id_torus],
                    "Neuron weight mismatch. Actual tuning weight %s,"
                    " but expected tuning weight is %s" %
                    (tuning_weight_vector[neuron - self.min_id_torus], tuning / float(num_stimulus_discr - 1))
                )

    def test_create_stimulus_based_local_connections(self):
        self.reset()

        r_loc = 0.2
        connect_dict = {"rule": "pairwise_bernoulli", "p": 0.8}
        inh_weight = -5.
        cap_s = 2.

        nc.create_stimulus_based_local_connections(
            self.torus_layer,
            self.torus_tree,
            self.neuron_to_tuning_map,
            self.tuning_to_neuron_map,
            self.inh_nodes,
            r_loc=r_loc,
            inh_weight=inh_weight,
            connect_dict=connect_dict
        )

        connect = nest.GetConnections(self.torus_layer)
        for c in connect:
            s = nest.GetStatus(c, "source")
            t = nest.GetStatus(c, "target")
            w = nest.GetStatus([c], "weight")[0]

            self.assertLessEqual(tp.Distance(s, t), r_loc)
            self.assertEqual(self.neuron_to_tuning_map[s], self.neuron_to_tuning_map[t])
            if s in self.inh_nodes:
                self.assertEqual(w, inh_weight, "Inhibitory weight was not set properly")
            else:
                self.assertEqual(w, cap_s, "Excitatory weight was not set properly")

    def test_create_stimulus_based_patches_random(self):
        self.reset()

        num_patches = 2
        r_loc = 0.2
        p_loc = 0.1
        p_p = 0.4
        connect_dict = {"rule": "pairwise_bernoulli", "p": 0.7}
        cap_s = 2.
        filter_patches = True

        nc.create_stimulus_based_patches_random(
            self.torus_layer,
            self.neuron_to_tuning_map,
            self.tuning_to_neuron_map,
            self.inh_nodes,
            self.torus_tree,
            num_patches=num_patches,
            r_loc=r_loc,
            p_loc=p_loc,
            p_p=p_p,
            cap_s=cap_s,
            connect_dict=connect_dict,
            filter_patches=filter_patches
        )

        connect = nest.GetConnections(self.torus_nodes)
        for c in connect:
            s = nest.GetStatus([c], "source")[0]
            t = nest.GetStatus([c], "target")[0]
            if t in self.torus_nodes:
                d = tp.Distance([s], [t])[0]
                w = nest.GetStatus([c], "weight")[0]

                self.assertLessEqual(d, self.size_layer)
                self.assertGreaterEqual(d, r_loc)
                self.assertEqual(self.neuron_to_tuning_map[s], self.neuron_to_tuning_map[t])
                self.assertEqual(w, cap_s, "The weight for patchy connections wasn't set properly")

    def test_create_input_current_generator(self):
        self.reset()

        input_stimulus = cs.image_with_spatial_correlation(size_img=(50, 50), radius=3, num_circles=80)
        organise_on_grid = True
        retina = nc.create_input_current_generator(
            input_stimulus,
            organise_on_grid=organise_on_grid
        )

        receptors = nest.GetNodes(retina)[0]
        amplitude = nest.GetStatus(receptors, "amplitude")
        self.assertTrue(
            np.all(np.asarray(amplitude) == input_stimulus.reshape(-1)),
            "The input stimulus and the current in the retina don't match"
        )

    def test_create_connections_rf(self):
        self.reset()

        rf_size = (10, 10)
        synaptic_strength = 1.
        calc_error = True
        use_dc = False
        p_rf = 1.

        positions = tp.GetPosition(self.torus_nodes)
        rf_centers = [
            (
                (x / (self.size_layer / 2.)) * self.input_stimulus.shape[1] / 2.,
                (y / (self.size_layer / 2.)) * self.input_stimulus.shape[0] / 2.
            )
            for (x, y) in positions
        ]

        adj_mat, recon = nc.create_connections_rf(
            self.input_stimulus,
            self.torus_nodes,
            rf_centers,
            self.neuron_to_tuning_map,
            self.inh_nodes,
            rf_size=rf_size,
            p_rf=p_rf,
            synaptic_strength=synaptic_strength,
            total_num_target=self.num_neurons,
            target_layer_size=self.size_layer,
            calc_error=calc_error,
            use_dc=use_dc,
        )

        # Check for lower equal since there can be less neurons at the edges of the retina
        self.assertLessEqual(
            np.max((adj_mat > 0).sum(axis=0)),  # (adj_mat > 0) since it is technically no adj mat but a tranform matrix
            rf_size[0] * rf_size[1],
            "There are receptors to which no connections are established although they are in the receptive field"
        )

        img_vector = np.ones(self.input_stimulus.size + 1)
        img_vector[:-1] = self.input_stimulus.reshape(-1)
        img_vector[img_vector == 0] = np.finfo("float64").eps
        recon_transform = sr.direct_stimulus_reconstruction(img_vector.dot(adj_mat)[:-1], adj_mat)

        for res, exp in zip(recon.reshape(-1), recon_transform.reshape(-1)):
            self.assertAlmostEqual(res, exp, msg="The transformation through the feedforward matrix does not lead to a "
                                                 "similar reconstruction")

    def test_create_torus_layer_with_jitter(self):
        self.reset()

        # Must be number that has a square root in N
        num_neurons = 144
        jitter = 0.01
        neuron_type = "iaf_psc_delta"
        layer_size = 3.

        layer = nc.create_torus_layer_with_jitter(
            num_neurons=num_neurons,
            jitter=jitter,
            neuron_type=neuron_type,
            layer_size=layer_size
        )

        nodes = nest.GetNodes(layer)[0]
        self.assertEqual(len(nodes), num_neurons, "Not the right number of nodes")

        model = nest.GetStatus(nodes, "model")
        for m in model:
            self.assertEqual(m, neuron_type, "Wrong neuron type set")

        mod_size = layer_size - jitter * 2
        step_size = mod_size / float(np.sqrt(num_neurons))
        coordinate_scale = np.arange(-mod_size / 2., mod_size / 2., step_size)
        grid = [[x, y] for y in coordinate_scale for x in coordinate_scale]

        for n in nodes:
            pos = np.asarray(tp.GetPosition([n])[0])
            sorted(grid, key=lambda l: np.linalg.norm(np.asarray(l) - pos))
            self.assertLessEqual(pos[0], np.abs(grid[0][0] + jitter), "Distance in x to original position too high")
            self.assertLessEqual(pos[1], np.abs(grid[0][1] + jitter), "Distance in y to original position too high")

    def test_create_distant_np_connections(self):
        self.reset()

        r_loc = .2
        p_loc = .5
        p_p = 0.5

        nc.create_distant_np_connections(self.torus_layer, p_loc=p_loc, r_loc=r_loc, p_p=p_p)

        conn = nest.GetConnections(source=self.torus_nodes)
        conn = [c for c in conn if nest.GetStatus([c], "target")[0] in self.torus_nodes]
        for c in conn:
            s = nest.GetStatus([c], "source")[0]
            t = nest.GetStatus([c], "target")[0]
            d = tp.Distance([s], [t])[0]
            self.assertLessEqual(d, self.size_layer / 2., "Distance is too large")
            self.assertGreaterEqual(d, r_loc, "Distance is too low")

    def test_create_random_patches(self):
        self.reset()

        r_loc = 0.2
        p_loc = 0.8
        p_p = 0.6
        num_patches = 2
        d_p = r_loc
        cap_s = 1.

        nc.create_random_patches(
            self.torus_layer,
            self.inh_nodes,
            r_loc=r_loc,
            p_loc=p_loc,
            num_patches=num_patches,
            p_p=p_p,
            cap_s=cap_s,
            plot=False
        )

        for s in self.torus_nodes:
            conn = nest.GetConnections(source=[s])
            targets = [t for t in nest.GetStatus(conn, "target") if t in self.torus_nodes]
            patches = []
            for t in targets:
                distance_s_t = tp.Distance([s], [t])[0]
                self.assertGreaterEqual(distance_s_t, r_loc, "Target node is too close")
                patch_not_existent = True
                for idx, patch in enumerate(patches):
                    patch_not_existent = False
                    for p in patch:
                        d = tp.Distance([t], [p])[0]
                        if d > d_p:
                            patch_not_existent = True
                            break

                    if not patch_not_existent:
                        patches[idx].add(t)
                        patch_not_existent = False
                        break

                if patch_not_existent:
                    patches.append(set([t]))
                self.assertLessEqual(len(patches), num_patches, "Created too many patches")

    def test_create_overlapping_patches(self):
        self.reset()

        r_loc = 0.3
        p_loc = 0.5
        p_r = r_loc / 2.
        distance = .7
        num_patches = 2
        p_p = 0.3

        nc.create_overlapping_patches(
            self.torus_layer,
            r_loc=r_loc,
            p_loc=p_loc,
            distance=distance,
            num_patches=num_patches,
            p_p=p_p
        )

        anchors = [tu.to_coordinates(n * 360. / float(num_patches), distance) for n in range(1, num_patches + 1)]

        for s in self.torus_nodes:
            pos_s = tp.GetPosition([s])[0]
            patch_centers = (np.asarray(anchors) + np.asarray(pos_s)).tolist()
            conn = nest.GetConnections(source=[s])
            targets = [t for t in nest.GetStatus(conn, "target") if t in self.torus_nodes]
            for t in targets:
                d = tp.Distance([s], [t])[0]
                self.assertGreaterEqual(d, distance - p_r, "Established connection is too short")
                self.assertLessEqual(d, distance + p_r, "Established connection is too long")
                d_to_centers = list(tp.Distance(patch_centers, [t]))
                self.assertTrue(
                    np.any(np.asarray(d_to_centers) <= p_r),
                    "Node is too far from any patch center"
                )

    def test_create_shared_patches(self):
        self.reset()

        r_loc = 0.1
        p_loc = 0.9
        p_p = 0.3
        d_p = r_loc + 0.1
        size_boxes = 0.2
        num_patches = 2
        num_shared_patches = 3

        nc.create_shared_patches(
            self.torus_layer,
            r_loc=r_loc,
            p_loc=p_loc,
            size_boxes=size_boxes,
            num_patches=num_patches,
            num_shared_patches=num_shared_patches,
            p_p=p_p
        )

        sublayer_anchors, box_mask = nc.create_distinct_sublayer_boxes(size_boxes, size_layer=self.size_layer)

        all_nodes = []
        for anchor in sublayer_anchors:
            box_nodes = tp.SelectNodesByMask(
                self.torus_layer,
                anchor,
                mask_obj=tp.CreateMask("rectangular", specs=box_mask)
            )

            box_patches = []
            for s in box_nodes:
                self.assertNotIn(s, all_nodes, "Boxes are not mutually distinct")
                all_nodes.append(s)

                conn = nest.GetConnections(source=[s])
                targets = [t for t in nest.GetStatus(conn, "target") if t in self.torus_nodes]
                patch_idx = set()
                for t in targets:
                    distance_anchor_t = tp.Distance([anchor], [t])[0]
                    self.assertGreaterEqual(distance_anchor_t, r_loc, "Target node is too close to anchor")
                    patch_not_existent = True
                    for idx, patch in enumerate(box_patches):
                        patch_not_existent = False
                        ds = tp.Distance([t], list(patch))
                        if np.any(np.asarray(ds) > d_p):
                            patch_not_existent = True
                            continue

                        if not patch_not_existent:
                            patch_idx.add(idx)
                            box_patches[idx].add(t)
                            patch_not_existent = False
                            break

                        self.assertLessEqual(len(patch_idx), num_patches, "Created too many patches per neuron")

                    if patch_not_existent:
                        box_patches.append({t})

                    # Chose num shared patches + 1, as it can happen that patches are very close to each other
                    # -x-x-x- ==> If all patches (x) are right next to each other the algorithm can accidentally
                    # see the spaces in between (-) as patch as well. Then the maximum is one more than the
                    # num of shared patches
                    self.assertLessEqual(len(box_patches), num_shared_patches + 1, "Created too many patches per box")

    def test_create_partially_overlapping_patches(self):
        self.reset()

        r_loc = 0.1
        p_loc = 0.7
        d_p = r_loc
        size_boxes = 0.2
        num_patches = 2
        num_shared_patches = 3
        num_patches_replaced = 3
        p_p = 0.3

        nc.create_partially_overlapping_patches(
            self.torus_layer,
            r_loc=r_loc,
            p_loc=p_loc,
            size_boxes=size_boxes,
            num_patches=num_patches,
            num_shared_patches=num_shared_patches,
            num_patches_replaced=num_patches_replaced,
            p_p=p_p
        )

        sublayer_anchors, box_mask = nc.create_distinct_sublayer_boxes(size_boxes, size_layer=self.size_layer)

        for anchor in sublayer_anchors:
            box_nodes = tp.SelectNodesByMask(
                self.torus_layer,
                anchor,
                mask_obj=tp.CreateMask("rectangular", specs=box_mask)
            )

            box_patches = []
            for s in box_nodes:
                conn = nest.GetConnections(source=[s])
                targets = [t for t in nest.GetStatus(conn, "target") if t in self.torus_nodes]
                patch_idx = set()
                for t in targets:
                    distance_anchor_t = tp.Distance([anchor], [t])[0]
                    self.assertGreaterEqual(distance_anchor_t, r_loc, "Target node is too close to anchor")
                    patch_not_existent = True
                    for idx, patch in enumerate(box_patches):
                        patch_not_existent = False
                        ds = tp.Distance([t], list(patch))
                        if np.any(np.asarray(ds) > d_p):
                            patch_not_existent = True
                            continue

                        if not patch_not_existent:
                            patch_idx.add(idx)
                            box_patches[idx].add(t)
                            patch_not_existent = False
                            break

                        self.assertLessEqual(len(patch_idx), num_patches, "Created too many patches per neuron")

                    if patch_not_existent:
                        box_patches.append({t})

                    # Chose num shared patches + 1, as it can happen that patches are very close to each other
                    # -x-x-x- ==> If all patches (x) are right next to each other the algorithm can accidentally
                    # see the spaces in between (-) as patch as well. Then the maximum is one more than the
                    # num of shared patches
                    self.assertLessEqual(len(box_patches), num_shared_patches+1, "Created too many patches per box")

    def test_a_create_perlin_stimulus_map(self):
        self.reset()

        num_stimulus_discr = 4
        resolution = (15, 15)
        spacing = 0.1
        plot = False
        save_plot = False

        tuning_to_neuron, neuron_to_tuning, color_map = nc.create_perlin_stimulus_map(
            self.torus_layer,
            self.inh_nodes,
            num_stimulus_discr=num_stimulus_discr,
            resolution=resolution,
            spacing=spacing,
            plot=plot,
            save_plot=save_plot
        )

        for n in self.torus_nodes:
            if n not in self.torus_nodes:
                self.assertIn(n, tuning_to_neuron[neuron_to_tuning[n]], "Neuron is not assigned to the correct tuning list")
                p = tp.GetPosition([n])[0]
                x_grid, y_grid = tu.coordinates_to_cmap_index(self.size_layer, p, spacing)
                self.assertEqual(
                    color_map[x_grid, y_grid],
                    neuron_to_tuning[n],
                    "Color map and tuning preference doesn't match"
                )

    def test_create_random_connections(self):
        self.reset()

        connect_dict = {
            "rule": "pairwise_bernoulli",
            "p": 0.5
        }
        cap_s = 2.
        inh_weight = -5.

        nc.create_random_connections(
            self.torus_layer,
            self.inh_nodes.tolist(),
            connect_dict=connect_dict,
            cap_s=cap_s,
            inh_weight=inh_weight
        )

        for n in self.torus_nodes:
            connect = nest.GetConnections(source=[n], target=self.torus_nodes)
            targets = nest.GetStatus(connect, "target")
            weights = nest.GetStatus(connect, "weight")
            if n in self.inh_nodes:
                self.assertTrue(np.all(np.asarray(weights) == inh_weight), "Inhibitory weights"
                                                                                 " were not set properly")
            else:
                in_tuning_class = [t in self.tuning_to_neuron_map[self.neuron_to_tuning_map[n]] for t in targets]
                self.assertFalse(np.all(np.asarray(in_tuning_class)),
                                 "With a random connection rule the connections must "
                                 "not be established solely to the same tuning class")
                self.assertTrue(np.all(np.asarray(weights) == cap_s), "Excitatory weights were not set properly")

    def test_create_local_circular_connections_topology(self):
        self.reset()

        r_loc = 0.3
        p_loc = 0.4

        nc.create_local_circular_connections_topology(self.torus_layer, r_loc=r_loc, p_loc=p_loc)
        connect = nest.GetConnections(self.torus_nodes)
        for c in connect:
            s = nest.GetStatus([c], "source")[0]
            t = nest.GetStatus([c], "target")[0]
            if t in self.torus_nodes:
                d = tp.Distance([s], [t])[0]
                self.assertLessEqual(d, r_loc)

    def test_set_synaptic_strength(self):
        self.reset()

        cap_s = 6.

        nc.create_random_connections(self.torus_layer, self.inh_nodes)
        adj_mat = na.create_adjacency_matrix(self.torus_nodes, self.torus_nodes)

        nc.set_synaptic_strength(
            self.torus_nodes,
            adj_mat,
            cap_s=cap_s,
            divide_by_num_connect=False
        )

        weights = nest.GetStatus(nest.GetConnections(source=self.torus_nodes, target=self.torus_nodes), "weight")
        self.assertTrue(np.all(np.asarray(weights) == cap_s), "Weights aren't set properly when not dividing by "
                                                              "number of neurons")

        nc.set_synaptic_strength(
            self.torus_nodes,
            adj_mat,
            cap_s=cap_s,
            divide_by_num_connect=True
        )

        weights = nest.GetStatus(nest.GetConnections(source=self.torus_nodes, target=self.torus_nodes), "weight")
        self.assertTrue(np.all(np.asarray(weights) == cap_s / adj_mat.sum()), "Weights aren't set properly when "
                                                                              "dividing by number of neurons")


    def test_create_sensory_nodes(self):
        self.reset()
        num_neurons = 1e3
        time_const = 20.0
        rest_pot = -70.
        threshold_pot = -55.
        capacitance = 80.

        nodes, _, _ = nc.create_sensory_nodes(
            num_neurons=num_neurons,
            time_const=time_const,
            rest_pot=rest_pot,
            threshold_pot=threshold_pot,
            capacitance=capacitance,
            use_barranca=False
        )

        self.assertEqual(len(nodes), num_neurons, "Didn't create the correct number of neurons")

        model = nest.GetStatus(nodes, "model")
        is_model = [m == "iaf_psc_delta" for m in model]
        self.assertTrue(np.all(np.asarray(is_model)), "Model was not set correctly")

        tau = nest.GetStatus(nodes, "tau_m")
        self.assertTrue(np.all(np.asarray(tau) == time_const), "Time constant was not set correctly")

        v_rest = nest.GetStatus(nodes, "E_L")
        self.assertTrue(np.all(np.asarray(v_rest) == rest_pot), "Resting potential was not set correctly")

        v_rest = nest.GetStatus(nodes, "V_reset")
        self.assertTrue(np.all(np.asarray(v_rest) == rest_pot), "Resting potential was not set correctly")

        v_th = nest.GetStatus(nodes, "V_th")
        self.assertTrue(np.all(np.asarray(v_th) == threshold_pot), "Threshold potential was not set correctly")

        cap = nest.GetStatus(nodes, "C_m")
        self.assertTrue(np.all(np.asarray(cap) == capacitance), "Capacitance was not set correctly")

        # TODO Comment in if barranca node model is installed
        # nodes, _, _ = nc.create_sensory_nodes(
        #     num_neurons=num_neurons,
        #     time_const=time_const,
        #     rest_pot=rest_pot,
        #     threshold_pot=threshold_pot,
        #     capacitance=capacitance,
        #     use_barranca=True
        # )
        #
        # self.assertEqual(len(nodes), num_neurons, "Didn't create the correct number of neurons")
        #
        # model = nest.GetStatus(nodes, "model")
        # is_model = [m == "barranca_neuron" for m in model]
        # self.assertTrue(np.all(np.asarray(is_model)), "Model was not set correctly")
        #
        # tau = nest.GetStatus(nodes, "tau_m")
        # self.assertTrue(np.all(np.asarray(tau) == time_const), "Time constant was not set correctly")
        #
        # v_rest = nest.GetStatus(nodes, "V_R")
        # self.assertTrue(np.all(np.asarray(v_rest) == rest_pot), "Resting potential was not set correctly")
        #
        # v_th = nest.GetStatus(nodes, "V_th")
        # self.assertTrue(np.all(np.asarray(v_th) == threshold_pot), "Threshold potential was not set correctly")
        #
        # cap = nest.GetStatus(nodes, "C_m")
        # self.assertTrue(np.all(np.asarray(cap) == capacitance), "Capacitance was not set correctly")

    def test_step_tuning_curve(self):
        input_stimulus = np.asarray([0, 1, 254, 255])
        expeted_class_0 = np.asarray([True, True, False, False])
        expeted_class_1 = np.asarray([False, False, True, True])
        tuning_discr_steps = 256 / 2.
        multiplier = 1.

        tuned_0 = nc.step_tuning_curve(
            input_stimulus=input_stimulus,
            stimulus_tuning=0,
            tuning_discr_steps=tuning_discr_steps,
            multiplier=multiplier
        )

        tuned_1 = nc.step_tuning_curve(
            input_stimulus=input_stimulus,
            stimulus_tuning=1,
            tuning_discr_steps=tuning_discr_steps,
            multiplier=multiplier
        )

        self.assertTrue(np.all(tuned_0 == expeted_class_0), "Input has not been converted properly for tuning class 0")
        self.assertTrue(np.all(tuned_1 == expeted_class_1), "Input has not been converted properly for tuning class 1")

    def test_continuous_tuning_curve(self):
        input_stimulus = np.asarray([0, 0, 256 / 2., 256 / 2.])
        expected_0 = np.asarray([255., 255., 154.66531823, 154.66531823])
        expected_1 = np.asarray([154.66531823, 154.66531823, 255., 255.])
        tuning_discr_steps = 256 / 2.
        max_value = 255.

        tuning_0 = nc.continuous_tuning_curve(
            input_stimulus=input_stimulus,
            stimulus_tuning=0,
            tuning_discr_steps=tuning_discr_steps,
            max_value=max_value
        )

        tuning_1 = nc.continuous_tuning_curve(
            input_stimulus=input_stimulus,
            stimulus_tuning=1,
            tuning_discr_steps=tuning_discr_steps,
            max_value=max_value
        )

        for tun0, ex0, tun1, ex1 in zip(tuning_0, expected_0, tuning_1, expected_1):
            self.assertAlmostEqual(tun0, ex0, msg="Input was not properly transformed for stimulus class 0")
            self.assertAlmostEqual(tun1, ex1, msg="Input was not properly transformed for stimulus class 1")

    def test_linear_tuning(self):
        input_stimulus = np.asarray([0, 1, 254, 255])
        exp_0 = np.asarray([0, 1, 254, 255])
        exp_1 = np.asarray([256, 255, 2, 1])
        exp_slope_0 = 1.
        exp_slope_1 = -1.
        exp_intercept_0 = 0.0
        exp_intercept_1 = 256.
        tuning_discr_steps = 256 / 2.

        tuning_0, slope_0, intercept_0 = nc.linear_tuning(
            input_stimulus=input_stimulus,
            stimulus_tuning=0,
            tuning_discr_steps=tuning_discr_steps
        )

        tuning_1, slope_1, intercept_1 = nc.linear_tuning(
            input_stimulus=input_stimulus,
            stimulus_tuning=1,
            tuning_discr_steps=tuning_discr_steps
        )

        self.assertEqual(slope_0, exp_slope_0, "Slope was not set correctly for tuning class 0")
        self.assertEqual(slope_1, exp_slope_1, "Slope was not set correctly for tuning class 1")
        self.assertEqual(intercept_0, exp_intercept_0, "Intercept was not set correctly for tuning class 0")
        self.assertEqual(intercept_1, exp_intercept_1, "Intercept was not set correctly for tuning class 1")
        self.assertTrue(np.all(tuning_0 == exp_0), "Input was not transformed correctly for tuning class 0")
        self.assertTrue(np.all(tuning_1 == exp_1), "Input was not transformed correctly for tuning class 1")

    def test_same_input_current(self):
        self.reset()

        synaptic_strength = 2.
        connect_prob = 1.
        value = 255 / 2.
        rf_size = (10, 10)

        generators = nc.same_input_current(
            layer= self.torus_layer,
            synaptic_strength=synaptic_strength,
            connect_prob=connect_prob,
            value=value,
            rf_size=rf_size,
            use_dc=True
        )

        for gen in generators:
            target = nest.GetStatus(nest.GetConnections(source=[gen]), "target")
            self.assertEqual(nest.GetStatus([gen], "model")[0], "dc_generator", "Generator is not a DC input generator")
            self.assertEqual(len(target), 1, "Generator is connected to several input neurons")
            self.assertIn(target[0], self.torus_nodes, "Generator is not connected to torus node")

        self.reset()

        generators = nc.same_input_current(
            layer=self.torus_layer,
            synaptic_strength=synaptic_strength,
            connect_prob=connect_prob,
            value=value,
            rf_size=rf_size,
            use_dc=False
        )

        for gen in generators:
            target = nest.GetStatus(nest.GetConnections(source=[gen]), "target")
            self.assertEqual(nest.GetStatus([gen], "model")[0], "poisson_generator", "Generator is not "
                                                                                     "a DC input generator")
            self.assertEqual(len(target), 1, "Generator is connected to several input neurons")
            self.assertIn(target[0], self.torus_nodes, "Generator is not connected to torus node")

    def test_convert_step_tuning(self):
        self.reset()

        rf = np.asarray([0, 1, 254, 255])
        indices = np.arange(0, 4)
        exp_0 = np.asarray([255., 255., 0., 0.])
        exp_1 = np.asarray([0., 0., 255., 255.])
        exp_adj_0 = np.asarray([255. / (np.finfo("float64").eps), 255. / (1. + np.finfo("float64").eps), 0., 0.])
        exp_adj_1 = np.asarray([0., 0., 255. / (254 + np.finfo("float64").eps), 255. / (255. + np.finfo("float64").eps)])
        tuning_discr_steps = 256 / 2.
        adj_mat = np.zeros((4, self.num_neurons))

        amplitude_0 = nc.convert_step_tuning(
            self.torus_nodes[0],
            rf=rf,
            neuron_tuning=0,
            tuning_discr_step=tuning_discr_steps,
            indices=indices,
            adj_mat=adj_mat,
            min_target=self.min_id_torus
        )

        amplitude_1 = nc.convert_step_tuning(
            self.torus_nodes[1],
            rf=rf,
            neuron_tuning=1,
            tuning_discr_step=tuning_discr_steps,
            indices=indices,
            adj_mat=adj_mat,
            min_target=self.min_id_torus
        )

        self.assertTrue(np.all(amplitude_0 == exp_0), "Amplitude was not set correctly for tuning class 0")
        self.assertTrue(np.all(amplitude_1 == exp_1), "Amplitude was not set correctly for tuning class 1")
        self.assertTrue(np.all(adj_mat[:, 0] == exp_adj_0), "Transformation matrix was not set properly for "
                                                            "tuning class 0")
        self.assertTrue(np.all(adj_mat[:, 1] == exp_adj_1), "Transformation matrix was not set properly for "
                                                            "tuning class 1")
        self.assertEqual(adj_mat[:, 2:].sum(), 0.0,  "Wrong values were set in the Transformation matrix")

    def test_convert_gauss_tuning(self):
        self.reset()

        rf = np.asarray([0, 0, 256 / 2., 256 / 2.])
        exp_0 = np.asarray([255., 255., 154.66531823, 154.66531823])
        exp_1 = np.asarray([154.66531823, 154.66531823, 255., 255.])
        indices = np.arange(0, 4)
        tuning_discr_steps = 256 / 2.
        adj_mat = np.zeros((4, self.num_neurons))

        amplitude_0 = nc.convert_gauss_tuning(
            self.torus_nodes[0],
            rf=rf,
            neuron_tuning=0,
            tuning_discr_step=tuning_discr_steps,
            indices=indices,
            adj_mat=adj_mat,
            min_target=self.min_id_torus
        )

        amplitude_1 = nc.convert_gauss_tuning(
            self.torus_nodes[1],
            rf=rf,
            neuron_tuning=1,
            tuning_discr_step=tuning_discr_steps,
            indices=indices,
            adj_mat=adj_mat,
            min_target=self.min_id_torus
        )

        self.assertTrue(np.all(np.abs(amplitude_0 - exp_0) < 1e-8),
                        "Amplitude was not set correctly for tuning class 0")
        self.assertTrue(np.all(np.abs(amplitude_1 - exp_1) < 1e-8),
                        "Amplitude was not set correctly for tuning class 1")

        rf[rf == 0] = np.finfo("float64").eps
        exp_adj_0 = exp_0 / rf
        exp_adj_1 = exp_1 / rf

        self.assertTrue(np.all(np.abs(adj_mat[:, 0] - exp_adj_0) < 1e-8),
                        "Transformation matrix was not set properly for tuning class 0")
        self.assertTrue(np.all(np.abs(adj_mat[:, 1] - exp_adj_1 < 1e-8)),
                        "Transformation matrix was not set properly for tuning class 1")
        self.assertEqual(adj_mat[:, 2:].sum(), 0.0,  "Wrong values were set in the Transformation matrix")

    def test_convert_linear_tuning(self):
        self.reset()

        rf = np.asarray([0, 1, 254, 255])
        exp_0 = np.asarray([0, 1, 254, 255])
        exp_1 = np.asarray([256, 255, 2, 1])
        indices = np.arange(0, 4)
        exp_slope_0 = 1.
        exp_slope_1 = -1.
        exp_intercept_0 = 0.0
        exp_intercept_1 = 256.
        tuning_discr_steps = 256 / 2.

        adj_mat = np.zeros((5, self.num_neurons))

        amplitude_0 = nc.convert_linear_tuning(
            self.torus_nodes[0],
            rf=rf,
            neuron_tuning=0,
            tuning_discr_step=tuning_discr_steps,
            indices=indices,
            adj_mat=adj_mat,
            min_target=self.min_id_torus
        )

        amplitude_1 = nc.convert_linear_tuning(
            self.torus_nodes[1],
            rf=rf,
            neuron_tuning=1,
            tuning_discr_step=tuning_discr_steps,
            indices=indices,
            adj_mat=adj_mat,
            min_target=self.min_id_torus
        )

        self.assertTrue(np.all(np.abs(amplitude_0 - exp_0) < 1e-8), "Amplitude was not set correctly "
                                                                    "for tuning class 0")
        self.assertTrue(np.all(np.abs(amplitude_1 - exp_1) < 1e-8), "Amplitude was not set correctly "
                                                                    "for tuning class 1")

        self.assertTrue(np.all(np.abs(adj_mat[:-1, 0] - exp_slope_0) < 1e-8),
                        "Slope was not set properly for tuning class 0")
        self.assertTrue(np.all(np.abs(adj_mat[:-1, 1] - exp_slope_1 < 1e-8)),
                        "Slope was not set properly for tuning class 1")
        self.assertTrue(np.all(np.abs(adj_mat[-1, 0] - rf.size * exp_intercept_0) < 1e-8),
                        "Intercept was not set properly for tuning class 0")
        self.assertTrue(np.all(np.abs(adj_mat[-1, 1] - rf.size * exp_intercept_1 < 1e-8)),
                        "Intercept was not set properly for tuning class 1")
        self.assertEqual(adj_mat[:, 2:].sum(), 0.0, "Wrong values were set in the Transformation matrix")

    def test_get_local_connectivity(self):
        r_loc = 0.5
        p_loc = 1.
        layer_size = 1
        exp = 0.7853981633974483
        local_conn, _ = nc.get_local_connectivity(r_loc=r_loc, p_loc=p_loc, layer_size=layer_size)

        self.assertAlmostEqual(local_conn, exp, msg="Local connectivity was not computed properly")

    def test_get_lr_connection_probability_patches(self):
        r_loc = 0.5
        p_loc = 1.
        r_p = 0.2
        num_patches = 3
        layer_size = 1.
        exp = 0.0

        lr_conn = nc.get_lr_connection_probability_patches(
            r_loc=r_loc,
            p_loc=p_loc,
            r_p=r_p,
            num_patches=num_patches,
            layer_size=layer_size
        )

        self.assertAlmostEqual(lr_conn, exp, msg="Patchy connectivity was not computed properly")

    def test_get_lr_connection_probability_np(self):
        r_loc = 0.3
        p_loc = .8
        layer_size = 1.
        exp = 0.0

        lr_conn = nc.get_lr_connection_probability_np(
            r_loc=r_loc,
            p_loc=p_loc,
            layer_size=layer_size
        )

        self.assertAlmostEqual(lr_conn, exp, msg="Patchy connectivity was not computed properly")


if __name__ == '__main__':
    unittest.main()
