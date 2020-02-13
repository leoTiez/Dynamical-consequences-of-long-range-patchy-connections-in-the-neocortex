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


def plot_connections(src_nodes, target_nodes, layer_size, save_plot=False, plot_name=None, color_mask=None):
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
    else:
        plt.show()


