#!/usr/bin/python3
import os
import argparse
from pathlib import Path
import numpy as np
from scipy.fftpack import idct
import scipy.interpolate as ip
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import nest
import nest.topology as tp


def arg_parse_plts():
    """
    Command line argument parser for the plotting file
    :return: The command line arguments
    """
    parser = argparse.ArgumentParser(description="Accept command line parameters for the creation of plots.")
    parser.add_argument("--type", type=str, help="Defines plot type. Can be either bar or gauss")
    parser.add_argument("--show", dest="show", action="store_true", help="Show plots instead of saving them")
    parser.add_argument("--path_full", type=str, help="Path to reconstruction error files with full sampling")
    parser.add_argument("--path_part", type=str, help="Path to reconstruction error files with subsampling")
    parser.add_argument("--network", type=str, help="Defines the network type")
    parser.add_argument("--input", type=str, help="Defines the input stimulus type")
    parser.add_argument("--tuning", type=str, help="Defines the tuning function")

    parsed_args = parser.parse_args()

    return parsed_args


def arg_parse():
    """
    Command line argument parser
    :return: The command line arguments
    """
    parser = argparse.ArgumentParser(description="Accept command line parameters for the reconstruction tests.")
    parser.add_argument("--agg", dest="agg", action="store_true", help="Use Agg backend for matplotlib")
    parser.add_argument("--seed", dest="seed", action="store_true", help="Seed random number generator")
    parser.add_argument("--network", type=str, help="Defines the network type")
    parser.add_argument("--input", type=str, help="Defines the input stimulus type")
    parser.add_argument("--parameter", type=str, help="Defines the parameter that is manipulated during experimenting")
    parser.add_argument("--tuning", type=str, help="Defines the tuning function")
    parser.add_argument("--cluster", type=tuple, help="Defines the cluster size")
    parser.add_argument("--patches", type=int, help="Defines the number of patches")
    parser.add_argument("--num_trials", type=int, help="Sets the number of trials")
    parser.add_argument("--img_prop", type=float, help="Sets the sampling rate. Value between 0 and 1")
    parsed_args = parser.parse_args()

    return parsed_args


def custom_cmap(num_stimulus_discr=4, add_inh=False):
    """
    Custom color map for coloring the neurons and tuning map
    :param num_stimulus_discr: Number of stimulus feature classes that can be discriminated
    :param add_inh: Flag to determine whether inhibitory neurons are to be included as they don't have any
    stimulus tuning
    :return: The customised color map
    """
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


def load_image(name="boat50.png", path=None, **kwargs):
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
        save_prefix="",
        color_mask=None
):
    """
    Plotting function for connections between nodes for visualisation and debugging purposes
    :param src_nodes: Source nodes
    :param target_nodes: Target nodes
    :param layer_size: Size of the layer
    :param save_plot: Flag for saving the plot
    :param plot_name: Name of the saved plot file. Is not taken into account if save_plot is False
    :param save_prefix: Naming prefix that is used if the plot save_plot is set to true
    :param color_mask: Color mask for the color/orientation map of neurons. If none it is not taken into account
    :return None
    """
    plt.axis((-layer_size/2., layer_size/2., -layer_size/2., layer_size/2.))
    source_positions = tp.GetPosition(src_nodes)
    x_source, y_source = zip(*source_positions)
    plt.plot(x_source, y_source, 'o')

    if len(target_nodes) > 0:
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
        Path(curr_dir + "/figures/connections/").mkdir(exist_ok=True, parents=True)
        plt.savefig(curr_dir + "/figures/connections/%s_%s" % (save_prefix, plot_name))
        plt.close()
    else:
        plt.show()


def plot_colorbar(fig, ax, num_stim_classes=4):
    """
    Adds a color bar to the plot
    :param fig: The matplotlib figure
    :param ax: The matplotlib axis
    :param num_stim_classes: The number of stimulus feature classes that can be discriminated
    :return: None
    """
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
    """
    Creates a color map using a Perlin noise distribution
    :param size_layer: Size of the layer
    :param resolution: The resolution of the Perlin noise defines the mesh on which values and vectors are randomly
    sampled
    :param spacing: The spacing in x and y for all values that are interpolated
    :return: The Perlin noise color map
    """
    grid_nodes_range = np.arange(0, size_layer, spacing)
    stimulus_grid_range_x = np.linspace(0, size_layer, resolution[0])
    stimulus_grid_range_y = np.linspace(0, size_layer, resolution[1])
    V = np.random.rand(stimulus_grid_range_x.size, stimulus_grid_range_y.size)

    ipol = ip.RectBivariateSpline(stimulus_grid_range_x, stimulus_grid_range_y, V)
    c_map = ipol(grid_nodes_range, grid_nodes_range)
    return c_map


def coordinates_to_cmap_index(layer_size, position, spacing):
    """
    Converts coordinates to an index in the tuning color map
    :param layer_size: Size of the layer
    :param position: Postion of the node
    :param spacing: Spacing that was used for creating the cmap
    :return: The indices of these coordinates for the color map
    """
    position = np.asarray(position)
    if len(position.shape) <= 1:
        position = np.asarray([position])
    y = np.floor(((layer_size / 2.) + position[:, 0]) / spacing).astype('int')
    x = np.floor(((layer_size / 2.) + position[:, 1]) / spacing).astype('int')

    return x, y


def dot_product_perlin(x_grid, y_grid, x, y, gradients):
    """
    The dot product step to compute the perlin noise
    :param x_grid: The closest x value on the grid
    :param y_grid: The closest y value on the grid
    :param x: The x value
    :param y: The y value
    :param gradients: The sampled gradients
    :return: The dot product for producing the Perlin noise
    """
    x_weight = x - x_grid
    y_weight = y - y_grid
    return x_weight * gradients[x_grid, y_grid][0] + y_weight * gradients[x_grid, y_grid][1]


def lerp_perlin(a, b, weight):
    """
    Function to linearly interpolate between a and b
    :param a: Point a
    :param b: Point b
    :param weight: Weigth between 0 and 1
    :return: Interpolation value
    """
    return (1. - weight) * a + weight * b


def sort_nodes_space(nodes, axis=0):
    """
    Sorting the nodes with respect to space and in either x or y direction
    :param nodes: The nodes that need to be sorted
    :param axis: Integer for the axis. 0 is x axis, 1 is y axis
    :return: The sorted list of nodes, the sorted list of positions
    """
    pos = tp.GetPosition(nodes)
    nodes_pos = list(zip(nodes, pos))
    if axis == 0:
        nodes_pos.sort(key=lambda p: (p[1][0], p[1][1]))
    elif axis == 1:
        nodes_pos.sort(key=lambda p: (p[1][1], p[1][0]))
    nodes, pos = zip(*nodes_pos)
    return nodes, pos


def get_in_out_degree(nodes, node_tree=None, node_pos=None, r_loc=0.5, r_p=None, size_layer=8.):
    """
    Computes the distribution of in- and outdegree in the network
    :param nodes: The network nodes
    :param node_tree: The positions of the nodes organised in a tree
    :param node_pos: The nodes positions
    :param r_loc: The local radius
    :param r_p: The patchy radius. If set to None the half of the local radius is taken
    :param size_layer: The size of the layer
    :return: indegree dist of all connections, outdegree dist of all connections, indegree dist of local connections,
    the outdegree dist of local connections, the indegree of long-range connections, the outdegree of long-range
    conncetions
    """
    in_degree = []
    out_degree = []

    in_degree_loc = []
    out_degree_loc = []

    in_degree_lr = []
    out_degree_lr = []

    min_id = min(nodes)
    for node in nodes:
        out_connect = nest.GetConnections(source=[node])
        in_connect = nest.GetConnections(target=[node])
        out_degree.append(len(out_connect))
        in_degree.append(len(in_connect))

        if node_tree is not None and node_pos is not None:
            out_partners = set(nest.GetStatus(out_connect, "target"))
            in_partners = set(nest.GetStatus(in_connect, "source"))
            pos = node_pos[node - min_id]

            if r_p is None:
                r_p = r_loc / 2.

            min_distance_lr = r_loc + r_p
            max_distance_lr = np.sqrt(size_layer ** 2 + size_layer ** 2) / 2. - r_p

            # Local in/out degree
            connect_partners = set((np.asarray(node_tree.query_ball_point(pos, r_loc)) + min_id).tolist())
            in_degree_loc.append(len(connect_partners.intersection(in_partners)))
            out_degree_loc.append(len(connect_partners.intersection(out_partners)))

            # Long range in/out degree
            inner_nodes = (np.asarray(node_tree.query_ball_point(pos, min_distance_lr)) + min_id).tolist()
            outer_nodes = (np.asarray(node_tree.query_ball_point(pos, max_distance_lr)) + min_id).tolist()
            patchy_partners = set(outer_nodes).difference(set(inner_nodes))
            in_degree_lr.append(len(patchy_partners.intersection(in_partners)))
            out_degree_lr.append(len(patchy_partners.intersection(out_partners)))

    return in_degree, out_degree, in_degree_loc, out_degree_loc, in_degree_lr, out_degree_lr


def plot_reconstruction(input_stimulus, reconstruction, save_plots=False, save_prefix=""):
    _, fig_2 = plt.subplots(1, 2, figsize=(10, 5))
    fig_2[0].imshow(reconstruction, cmap='gray')
    fig_2[1].imshow(input_stimulus, cmap='gray', vmin=0, vmax=255)
    if not save_plots:
        plt.show()
    else:
        curr_dir = os.getcwd()
        Path(curr_dir + "/figures/reconstruction").mkdir(parents=True, exist_ok=True)
        plt.savefig(curr_dir + "/figures/reconstruction/%s_reconstruction.png" % save_prefix)
        plt.close()

