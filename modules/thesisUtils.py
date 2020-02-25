#!/usr/bin/python3
import os
import numpy as np
from scipy.fft import idct
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import nest.topology as tp


def degree_to_rad(deg):
    """
    Convert degree to radians
    :param deg: value in degree
    :return: value in radians
    """
    return deg * np.pi / 180.


def to_coordinates(angle, distance):
    """
    Convert distance and angle to coordinates
    :param angle: angle
    :param distance: distance
    :return: coordinates [x, y]
    """
    angle = degree_to_rad(angle)
    return [distance * np.cos(angle), distance * np.sin(angle)]


def idct2(x):
    """
    Two dimensional inverse discrete cosine transform
    :param x: Input array
    :return: The two-dim array computed through the two-dim inverse discrete cosune transform
    """
    return idct(idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def load_image(name, path=None):
    """
    Load image with given name from path
    :param name: Name with suffix of the picture
    :param path: Path to the image. If None is passed, the current directory + '/test-input/' is taken
    :return: The image as numpy array
    """
    if path is None:
        path = os.getcwd() + "/test-input/"
    image = Image.open(path + name).convert("L")
    return np.asarray(image)


def plot_connections(
        src_nodes,
        target_nodes,
        layer_size,
        save_plot=False,
        plot_name=None,
        color_mask=None
):
    """
    Plotting function for connections between nodes for visualisation and debugging purposes
    :param src_nodes: Source nodes
    :param target_nodes: Target nodes
    :param layer_size: Size of the layer
    :param save_plot: Flag for saving the plot
    :param plot_name: Name of the saved plot file. Is not taken into account if save_plot is False
    :param color_mask: Color mask for the color/orientation map of neurons. If none it is not taken into account
    """
    plt.axis((-layer_size/2., layer_size/2., -layer_size/2., layer_size/2.))
    source_positions = tp.GetPosition(src_nodes)
    x_source, y_source = zip(*source_positions)
    plt.plot(x_source, y_source, 'o')

    target_positions = tp.GetPosition(target_nodes)
    x_target, y_target = zip(*target_positions)
    plt.plot(x_target, y_target, 'o')

    for s in source_positions:
        for t in target_positions:
            plt.plot([s[0], t[0]], [s[1], t[1]], color="k")

    if color_mask is not None:
        for mask in color_mask:
            area_rect = patches.Rectangle(
                mask["lower_left"],
                width=mask["width"],
                height=mask["height"],
                color=mask["color"],
                alpha=0.2
            )
            plt.gca().add_patch(area_rect)
    if plot_name is None:
        plot_name = "connections.png"
    if save_plot:
        curr_dir = os.getcwd()
        plt.savefig(curr_dir + "/figures/" + plot_name)
        plt.close()
    else:
        plt.show()


def dot_product_perlin(x_grid, y_grid, x, y, gradients):
    x_weight = x - x_grid
    y_weight = y - y_grid
    return x_weight * gradients[x_grid, y_grid][0] + y_weight * gradients[x_grid, y_grid][1]


def lerp_perlin(a, b, weight):
    return (1. - weight) * a + weight * b