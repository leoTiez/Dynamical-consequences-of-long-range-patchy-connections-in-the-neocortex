#!/usr/bin/python3

import unittest

import modules.networkAnalysis as na

import numpy as np
import nest
nest.set_verbosity("M_ERROR")


class NetworkAnalysisTest(unittest.TestCase):
    @staticmethod
    def reset():
        nest.ResetKernel()

    @staticmethod
    def create_network(num_nodes):
        nodes = nest.Create("iaf_psc_delta", num_nodes)
        nest.Connect([nodes[0]], [nodes[1]])
        nest.Connect([nodes[0]], [nodes[2]])
        nest.Connect([nodes[2]], [nodes[3]])
        conn = nest.GetConnections(nodes)

        return nodes, conn

    def test_set_values_in_adjacency_matrix(self):
        num_nodes = 4

        self.reset()

        nodes, conn = self.create_network(num_nodes)

        adj_mat = np.zeros((num_nodes, num_nodes))
        adj_mat = na.set_values_in_adjacency_matrix(
            conn,
            adj_mat,
            min(nodes),
            min(nodes),
        )

        self.assertEqual(adj_mat[0, 1], 1, "Adjacency has not been set between 0 and 1")
        self.assertEqual(adj_mat[0, 2], 1, "Adjacency has not been set between 0 and 2")
        self.assertEqual(adj_mat[2, 3], 1, "Adjacency has not been set between 2 and 3")

    def test_create_adjacency_matrix(self):
        num_nodes = 4

        self.reset()

        nodes, conn = self.create_network(num_nodes)

        adj_mat_exp = np.zeros((num_nodes, num_nodes))
        adj_mat_exp[0, 1] = 1
        adj_mat_exp[0, 2] = 1
        adj_mat_exp[2, 3] = 1

        adj_mat_comp = na.create_adjacency_matrix(nodes, nodes)

        self.assertTrue(np.all(adj_mat_exp == adj_mat_comp), "Expected and actual adjacency matrix do not match")

    def test_eigenvalue_analysis(self):
        self.reset()

        test_mat = np.asarray([[0, 1], [-2, -3]])
        eigenvalues_exp = {-1, -2}
        ratio_eigenvec = [-1, -0.5]

        eigvalues, eigvec = na.eigenvalue_analysis(test_mat, plot=False)

        self.assertSetEqual(eigenvalues_exp, set(eigvalues.tolist()), "Eigenvalues do not match")
        self.assertIn(eigvec[0][0] / eigvec[1][0], ratio_eigenvec, "Eigenvalues have wrong ratio")
        self.assertIn(eigvec[0][1] / eigvec[1][1], ratio_eigenvec, "Eigenvalues have wrong ratio")

    def test_get_firing_rates(self):
        self.reset()
        firing_times = np.asarray([0, 1, 1, 2, 1, 0, 2, 2, 0, 1])
        count_exp = [3., 4., 3.]
        dummy_nodes = nest.Create("iaf_psc_delta", 3)
        firing_times = min(dummy_nodes) + firing_times
        count = na.get_firing_rates(firing_times, dummy_nodes, 1000)

        self.assertListEqual(count_exp, count.tolist(), "Spike count does not match")

    def test_mutual_information_img(self):
        self.reset()
        input_data = np.zeros((30, 30))
        recon_data = np.ones((30, 30))

        mi = na.mutual_information_img(input_data, recon_data)

        self.assertEqual(mi, 0., "MI not computed correctly")

        input_data = np.random.randint(0, 256, size=(30, 30))
        recon_data = input_data.copy() * 40

        mi_self = na.mutual_information_img(input_data, input_data)
        mi_scaled = na.mutual_information_img(input_data, recon_data)
        self.assertEqual(mi_scaled, mi_self, "MI not same if rescaled")

    def test_mutual_information_hist(self):
        self.reset()
        input_data = np.random.randint(0, 256, size=(30, 30))
        firing_rates = np.random.choice(input_data.reshape(-1), size=(1, 100), replace=False)

        mi = na.mutual_information_hist(input_data, firing_rates)
        mi_2 = na.mutual_information_hist(input_data*2, firing_rates*2)
        mi_3 = na.mutual_information_hist(input_data, firing_rates*2)
        self.assertEqual(mi, mi_2, "MI should be the same if both is scaled by the same factor")
        self.assertNotEqual(mi, mi_3, "MI should not be the same if only one parameter is scaled")

    def test_error_distance(self):
        self.reset()
        original = np.ones(10)
        reconstruction = original * 2

        err = na.error_distance(original, reconstruction)
        self.assertEqual(err, 0.0, "The error when using vectors scaled by a scalar should be 0")

        reconstruction = original * 0
        err = na.error_distance(original, reconstruction)
        self.assertEqual(err, 1.0, "The error between the zero vector and the original one should be 1.0")


if __name__ == '__main__':
    unittest.main()
