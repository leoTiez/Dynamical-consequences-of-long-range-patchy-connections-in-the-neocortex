#!/usr/bin/python3
import os
import argparse
import numpy as np
from scipy.fftpack import idct
import scipy.interpolate as ip
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import nest.topology as tp


def arg_parse():
    parser = argparse.ArgumentParser(description='Accept command line parameters for the reconstruction tests.')
    parser.add_argument('--agg', dest='agg', action='store_true', help='Use Agg backend for matplotlib')
    parser.add_argument('--seed', dest='seed', action='store_true', help='Seed random number generator')
    parsed_args = parser.parse_args()

    return parsed_args


def custom_cmap(num_stimulus_discr=4, add_inh=False):
    cmap = plt.get_cmap('tab10')
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0.0, b=(1/num_stimulus_discr)+0.1),
        cmap(np.linspace(0.0, (1/num_stimulus_discr)+0.1, 100)))
    if add_inh:
        new_cmap.set_under('k')
    return new_cmap


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
        plot_colorbar(plt.gcf(), plt.gca(), num_stim_classes=color_mask.max()+1)
        if not type(color_mask) == np.ndarray:
            for mask in color_mask:
                area_rect = patches.Rectangle(
                    mask["lower_left"],
                    width=mask["width"],
                    height=mask["height"],
                    color=mask["color"],
                    alpha=0.4
                )
                plt.gca().add_patch(area_rect)
        else:
            plt.imshow(
                color_mask,
                origin=(color_mask.shape[0] // 2, color_mask.shape[1] // 2),
                extent=(-layer_size / 2., layer_size / 2., -layer_size / 2., layer_size / 2.),
                cmap=custom_cmap(color_mask.max()+1),
                alpha=0.4
            )
    if plot_name is None:
        plot_name = "connections.png"
    if save_plot:
        curr_dir = os.getcwd()
        plt.savefig(curr_dir + "/figures/" + plot_name)
        plt.close()
    else:
        plt.show()


def plot_colorbar(fig, ax, num_stim_classes=4):
    step_size = 255 / float(num_stim_classes)
    bounds = [i * step_size for i in range(0, num_stim_classes + 1)]
    cmap = custom_cmap(num_stimulus_discr=num_stim_classes, add_inh=True)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        extend='min',
        boundaries=[-1] + bounds,
        ticks=bounds,
        spacing='uniform',
        orientation='vertical'
    )


def perlin_noise(size_layer=50, resolution=(5, 5), spacing=0.01):
    grid_nodes_range = np.arange(0, size_layer, spacing)
    stimulus_grid_range_x = np.linspace(0, size_layer, resolution[0])
    stimulus_grid_range_y = np.linspace(0, size_layer, resolution[1])
    V = np.random.rand(stimulus_grid_range_x.size, stimulus_grid_range_y.size)

    ipol = ip.RectBivariateSpline(stimulus_grid_range_x, stimulus_grid_range_y, V)
    c_map = ipol(grid_nodes_range, grid_nodes_range)
    return c_map


def coordinates_to_cmap_index(layer_size, position, spacing):
    y = np.floor(((layer_size / 2.) + position[0]) / spacing).astype('int')
    x = np.floor(((layer_size / 2.) + position[1]) / spacing).astype('int')

    return x, y


def dot_product_perlin(x_grid, y_grid, x, y, gradients):
    x_weight = x - x_grid
    y_weight = y - y_grid
    return x_weight * gradients[x_grid, y_grid][0] + y_weight * gradients[x_grid, y_grid][1]


def lerp_perlin(a, b, weight):
    return (1. - weight) * a + weight * b

