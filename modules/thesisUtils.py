#!/usr/bin/python3
import os
import argparse
from pathlib import Path
import numpy as np
from scipy.fftpack import idct
import scipy.interpolate as ip
from PIL import Image
from webcolors import hex_to_rgb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import nest
import nest.topology as tp


def arg_parse_plts(args):
    """
    Command line argument parser for the plotting file
    :return: The command line arguments
    """
    parser = argparse.ArgumentParser(description="Accept command line parameters for the creation of plots.")
    parser.add_argument("--show", dest="show", action="store_true", help="Show plots instead of saving them")
    parser.add_argument("--path", type=str, help="Path to reconstruction error")
    parser.add_argument("--x", type=str, help="Independent variable that is plotted")
    parser.add_argument("--y", type=str, help="Dependent variable that is plotted")
    parser.add_argument("--group", type=str, help="Parameter by which the data is grouped")
    parser.add_argument("--network", type=str, help="Filters for network")
    parser.add_argument("--input", type=str, help="Filters for input")
    parser.add_argument("--experiment", type=str, help="Filters for experiment")
    parser.add_argument("--sampling", type=str, help="Filters for parameter")
    parser.add_argument("--parameter", type=str, help="Filters for experiment parameter")
    parser.add_argument("--measure", type=str, help="Filters for measurement")
    parser.add_argument("--title", type=str, help="Name of the plot")

    parsed_args = parser.parse_args(args)

    return parsed_args


def arg_parse(args):
    """
    Command line argument parser
    :return: The command line arguments
    """
    parser = argparse.ArgumentParser(description="Accept command line parameters for the reconstruction tests.")
    parser.add_argument("--agg", dest="agg", action="store_true", help="Use Agg backend for matplotlib")
    parser.add_argument("--seed", dest="seed", action="store_true", help="Seed random number generator")
    parser.add_argument("--show", dest="show", action="store_true", help="Show plots instead of saving them")
    parser.add_argument("--network", type=str, help="Defines the network type")
    parser.add_argument("--input", type=str, help="Defines the input stimulus type")
    parser.add_argument("--num_neurons", type=int, help="Defines the number of sensory neurons")
    parser.add_argument("--verbosity", type=int, help="Sets the verbosity flag")
    parser.add_argument("--parameter", type=str, help="Defines the parameter that is manipulated during experimenting")
    parser.add_argument("--tuning", type=str, help="Defines the tuning function")
    parser.add_argument("--cluster", type=int, help="Defines the cluster size")
    parser.add_argument("--patches", type=int, help="Defines the number of patches")
    parser.add_argument("--sum", dest="sum", action="store_true", help="Computes the sum of the input weights. "
                                                                       "Only used for ThesisEigenvalueSpec script")
    parser.add_argument("--num_trials", type=int, help="Sets the number of trials")
    parser.add_argument("--ff_factor", type=float, help="Sets the weight factor that is multiplied to the "
                                                        "default value of the feedforward weights")
    parser.add_argument("--img_prop", type=str, help="Sets the sampling rate. Value between 0 and 1")
    parser.add_argument("--normalise", dest="normalise", action="store_true", help="If set, the activity of the"
                                                                                   "matrix dynamics test is normalised")
    parser.add_argument("--spatial_sampling",
                        dest="spatial_sampling",
                        action="store_true",
                        help="If the flag is set, the neurons that receive ff input are chosen with a "
                             "spatial correlation")
    parser.add_argument("--equilibrium", dest="equilibrium", action="store_true", help="If set, only the last 400ms are "
                                                                                       "used for the reconstruction")

    parsed_args = parser.parse_args(args)

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


def firing_rate_sorting(idx_based_list, sorted_list, new_idx_neurons, element):
    """
    Assign new indices to neurons to plot them ascendingly  wrt their stimulus class
    :param idx_based_list: New indices of firing neurons that are sorted according to their stimulus class
    :param sorted_list: List of firing neurons sorted according to their stimulus class
    :param new_idx_neurons: Mapping of original neuron id to plotting neuron id
    :param element: The current element for which a new index is calculated
    :return: The new index of the 'element'
    """
    if len(idx_based_list) == 0:
        new_idx_neurons[element] = 0
    if element not in new_idx_neurons.keys():
        new_idx_neurons[element] = np.minimum(list(sorted_list).index(element), max(idx_based_list) + 1)
    return new_idx_neurons[element]


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


def plot_reconstruction(input_stimulus, reconstruction, save_plots=False, save_prefix=""):
    """
    Plot stimulus reconstruction
    :param input_stimulus: Original input stimulus
    :param reconstruction: Reconstructed stimulus
    :param save_plots: If set to true, the plot is saved
    :param save_prefix: Prefix that is used for the saved file to identify the plot and the corresponding experiment
    :return: None
    """
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


def plot_cmap(
        ff_nodes,
        inh_nodes,
        color_map,
        stim_class,
        positions,
        muted_nodes=[],
        size_layer=8.,
        resolution=(10, 10),
        num_stimulus_discr=4,
        save_plot=False,
        save_prefix=""
):
    """
    Plot the tuning map
    :param ff_nodes: Nodes that receive feedforward input
    :param inh_nodes: Inhibitory neurons
    :param color_map: The color map
    :param stim_class: Tuning classes of all neurons
    :param positions: Position of all neurons
    :param muted_nodes: Neurons without ff input
    :param size_layer: Size of the layer, ie length of one side of the square sheet
    :param resolution: Resolution that was used to create the color map
    :param num_stimulus_discr: Number of tuning classes
    :param save_plot: If set to true, the plot is saved
    :param save_prefix: Prefix that is used for the saved file to identify the plot and the corresponding experiment
    :return: None
    """
    stimulus_grid_range_x = np.linspace(0, size_layer, resolution[0])
    stimulus_grid_range_y = np.linspace(0, size_layer, resolution[1])
    plt.imshow(
        color_map,
        origin=(stimulus_grid_range_x.size // 2, stimulus_grid_range_y.size // 2),
        extent=(-size_layer / 2., size_layer / 2., -size_layer / 2., size_layer / 2.),
        cmap=custom_cmap(num_stimulus_discr),
        alpha=0.4
    )

    min_idx = np.minimum(min(ff_nodes), min(muted_nodes)) if len(muted_nodes) > 0 else min(ff_nodes)
    inh_mask = np.zeros(len(ff_nodes) + len(muted_nodes)).astype('bool')
    inh_mask[np.asarray(inh_nodes) - min_idx] = True
    c = np.full(len(ff_nodes) + len(muted_nodes), '#000000')
    c[~inh_mask] = np.asarray(list(mcolors.TABLEAU_COLORS.items()))[stim_class, 1]

    c_rgba = np.ones((len(ff_nodes) + len(muted_nodes), 4))
    for num, color in enumerate(c):
        c_rgba[num, :3] = np.asarray(hex_to_rgb(color))[:] / 255.

    c_rgba[np.asarray(muted_nodes).astype("int64") - min_idx, 3] = 0.2
    plt.scatter(
        np.asarray(positions)[:, 0],
        np.asarray(positions)[:, 1],
        c=c_rgba
    )

    if not save_plot:
        plt.show()
    else:
        curr_dir = os.getcwd()
        Path(curr_dir + "/figures/tuning-map/").mkdir(parents=True, exist_ok=True)
        plot_name = "ff_neurons.png"
        plt.savefig(curr_dir + "/figures/tuning-map/%s_%s" % (save_prefix, plot_name))
        plt.close()