#!/usr/bin/python3
import unittest

import modules.thesisUtils as tu
from modules.networkConstruction import create_torus_layer_with_jitter
import numpy as np

import nest
import nest.topology as tp
nest.set_verbosity("M_ERROR")


class ThesisUtilsTest(unittest.TestCase):
    @staticmethod
    def reset():
        nest.ResetKernel()

    def test_degree_to_rad(self):
        deg = 45
        rad_ex = 0.785398

        rad_comp = tu.degree_to_rad(deg)

        self.assertAlmostEqual(rad_comp, rad_ex, 6, "The computed radians are not almost equal")

    def test_to_coordinates(self):
        angle = 90
        distance = 4
        expected = [0, 4]
        coordinates = tu.to_coordinates(angle, distance)

        self.assertAlmostEqual(coordinates[0], expected[0], 10, "The x-coordinate is too far off")
        self.assertAlmostEqual(coordinates[1], expected[1], 10, "The x-coordinate is too far off")

    def test_arg_parse_plts(self):
        path = "/exp"
        x = "x_test"
        y = "y_test"
        group = "group_test"
        network = "network_test"
        stimulus = "input_test"
        exp = "experiment_test"
        sampling = "sampling_test"
        param = "parameter_test"
        measure = "measure_test"
        name = "name_test"

        args = [
            "--show",
            "--path=" + path,
            "--x=" + x,
            "--y=" + y,
            "--group=" + group,
            "--network=" + network,
            "--input=" + stimulus,
            "--experiment=" + exp,
            "--sampling=" + sampling,
            "--parameter=" + param,
            "--measure=" + measure,
            "--name=" + name
        ]

        cmd_params = tu.arg_parse_plts(args)

        self.assertEqual(cmd_params.path, path, "Didn't convert --path properly")
        self.assertEqual(cmd_params.x, x, "Didn't convert --x properly")
        self.assertEqual(cmd_params.y, y, "Didn't convert --y properly")
        self.assertEqual(cmd_params.group, group, "Didn't convert --group properly")
        self.assertEqual(cmd_params.network, network, "Didn't convert --network properly")
        self.assertEqual(cmd_params.input, stimulus, "Didn't convert --input properly")
        self.assertEqual(cmd_params.experiment, exp, "Didn't convert --experiment properly")
        self.assertEqual(cmd_params.sampling, sampling, "Didn't convert --sampling properly")
        self.assertEqual(cmd_params.parameter, param, "Didn't convert --parameter properly")
        self.assertEqual(cmd_params.measure, measure, "Didn't convert --measure properly")
        self.assertEqual(cmd_params.name, name, "Didn't convert --name properly")
        self.assertTrue(cmd_params.show, "Didn't convert --show properly")

    def test_arg_parse(self):
        network = "network_test"
        stimulus = "input_test"
        img_prop = "sampling_test"
        param = "parameter_test"
        num_neurons = 1000
        verbosity = 3
        tuning = "tuning_test"
        cluster = 4
        patches = 1
        num_trials = 5
        ff_weight = 0.1
        rec_weight = 0.1

        args = [
            "--agg",
            "--seed",
            "--show",
            "--spatial_sampling",
            "--network=" + network,
            "--input=" + stimulus,
            "--num_neurons=" + str(num_neurons),
            "--verbosity=" + str(verbosity),
            "--parameter=" + param,
            "--tuning=" + tuning,
            "--cluster=" + str(cluster),
            "--patches=" + str(patches),
            "--num_trials=" + str(num_trials),
            "--ff_weight=" + str(ff_weight),
            "--rec_weight=" + str(rec_weight),
            "--img_prop=" + str(img_prop)
        ]

        cmd_params = tu.arg_parse(args)
        self.assertTrue(cmd_params.agg, "Didn't convert --agg properly")
        self.assertTrue(cmd_params.show, "Didn't convert --show properly")
        self.assertTrue(cmd_params.seed, "Didn't convert --seed properly")
        self.assertTrue(cmd_params.spatial_sampling, "Didn't convert --spatial_sampling properly")
        self.assertEqual(cmd_params.network, network, "Didn't convert --network properly")
        self.assertEqual(cmd_params.input, stimulus, "Didn't convert --input properly")
        self.assertEqual(cmd_params.num_neurons, num_neurons, "Didn't convert --num_neurons properly")
        self.assertEqual(cmd_params.verbosity, verbosity, "Didn't convert --verbosity properly")
        self.assertEqual(cmd_params.parameter, param, "Didn't convert --parameter properly")
        self.assertEqual(cmd_params.tuning, tuning, "Didn't convert --tuning properly")
        self.assertEqual(cmd_params.cluster, cluster, "Didn't convert --cluster properly")
        self.assertEqual(cmd_params.patches, patches, "Didn't convert --patches properly")
        self.assertEqual(cmd_params.num_trials, num_trials, "Didn't convert --num_trials properly")
        self.assertEqual(cmd_params.ff_weight, ff_weight, "Didn't convert --ff_weight properly")
        self.assertEqual(cmd_params.rec_weight, rec_weight, "Didn't convert --rec_weight properly")
        self.assertEqual(cmd_params.img_prop, img_prop, "Didn't convert --img_prop properly")

    def test_custom_cmap(self):
        num_stimulus_discr = 2
        add_inh = True
        name = 'trunc({n},{a:.2f},{b:.2f})'.format(n="tab10", a=0.0, b=(1/num_stimulus_discr)+0.1)
        new_cmap = tu.custom_cmap(num_stimulus_discr=num_stimulus_discr, add_inh=add_inh)

        self.assertEqual(new_cmap.name, name, "Cmap name was not set properly")

    def test_idct2(self):
        x = np.ones(2)
        x_idct2 = tu.idct2(x)
        for x1, x2 in zip(x_idct2, x):
            self.assertAlmostEqual(x1, x2, msg="Two dimensional cosine transform doesn't work")

        y = np.ones(3)
        y_expect = np.asarray([0.9218757,  0.7498963,  1.26007966])
        y_idct2 = tu.idct2(y)
        for y1, y2 in zip(y_idct2, y_expect):
            self.assertAlmostEqual(y1, y2, msg="Two dimensional cosine transform doesn't work")

    def test_coordinates_to_cmap_index(self):
        layer_size = 2
        position = (0.95, 0.95)
        spacing = 0.1
        expected_x = 19
        expected_y = 19

        x, y = tu.coordinates_to_cmap_index(layer_size, position, spacing)

        self.assertEqual(x[0], expected_x, "Coordinate conversion doesn't work")
        self.assertEqual(y[0], expected_y, "Coordinate conversion doesn't work")

    def test_sort_nodes_space(self):
        self.reset()
        layer = create_torus_layer_with_jitter(4, jitter=0, layer_size=2.)
        expected_positions = [[-1., -1.], [-1., 0.], [0., -1.], [0., 0.]]
        sorted_nodes, positions = tu.sort_nodes_space(nest.GetNodes(layer)[0], axis=0)
        for sp, esp in zip(positions, expected_positions):
            self.assertListEqual(list(sp), esp, "Nodes were not correctly sorted")

    def test_get_in_out_degree(self):
        self.reset()
        layer = create_torus_layer_with_jitter(4, jitter=0, layer_size=2.)
        conn_dict = {
            "connection_type": "divergent",
            "mask": {"rectangular": {
                "lower_left": [-1.0, -1.0],
                "upper_right": [1.0, 1.0]
            }},
            "kernel": 1.0
        }

        expect_in = [9, 6, 6, 4]
        expect_out = [4,6, 6, 9]

        tp.ConnectLayers(layer, layer, conn_dict)
        in_degree, out_degree, _, _, _, _ = tu.get_in_out_degree(nest.GetNodes(layer)[0])
        self.assertListEqual(expect_in, in_degree, "In degree was not computed correctly")
        self.assertListEqual(expect_out, out_degree, "Out degree was not computed correctly")

    def test_firing_rate_sorting(self):
        firing = [2, 0, 2, 2, 0, 1]
        expected_reindexing = [0, 1, 0, 0, 1, 2]

        reindexing_sp = []
        reindexing_neurons = {}
        for f in firing:
            reindexing_sp.append(tu.firing_rate_sorting(reindexing_sp, firing, reindexing_neurons, f))

        self.assertListEqual(
            reindexing_sp,
            expected_reindexing,
            "The reindexing procedure doesn't match the expected outcome"
        )


if __name__ == '__main__':
    unittest.main()
