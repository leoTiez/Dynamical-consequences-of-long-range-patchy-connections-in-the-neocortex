#!/usr/bin/python
import unittest

import modules.thesisUtils as tu


class ThesisUtilsTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
