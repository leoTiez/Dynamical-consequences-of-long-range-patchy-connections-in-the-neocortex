#!/usr/bin/python3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from createThesisNetwork import NETWORK_TYPE
from modules.networkConstruction import TUNING_FUNCTION
from modules.thesisUtils import arg_parse_plts
from modules.createStimulus import INPUT_TYPE

PLOT_TYPE={
    "bar": 0,
    "gauss": 1,
}


def information_loss(error_full, error_part):
    return error_part - error_full


def main_error(
        path_full,
        path_part,
        network="",
        stimulus="",
        attributes=[""],
        plt_bottom_margin=0.2,
        plot_type=PLOT_TYPE["bar"],
        save_plot=False
):
    curr_dir = os.getcwd() + "/"
    variance_full_l = []
    variance_part_l = []
    mean_full_l = []
    mean_part_l = []
    inf_loss_l = []
    label_names = []
    part_files = os.listdir(curr_dir + path_part)
    for file_full in sorted(os.listdir(curr_dir + path_full)):
        is_mean_error = "mean_error.txt" in file_full
        cut_off_idx = -15 if is_mean_error else -19
        file_part = list(filter(lambda f: file_full[:cut_off_idx] in f, part_files))
        file_part = list(
            filter(lambda f: "mean_error.txt" in f if is_mean_error else "error_variance.txt" in f, file_part)
        )[0]

        split_full = file_full.split("_")
        split_part = file_part.split("_")
        is_attribute = np.all(np.asarray([a in file_full and a in file_part for a in attributes]))
        is_patchy = "patchy" in file_full and "patchy" in file_part
        if "patchy" in network:
            is_network = (network in file_full and network in file_part) and is_patchy
        elif "random" in network:
            is_network = network in split_full[1] and network in split_part[1]
        elif len(network) == 0:
            is_network = True
        else:
            is_network = (network in file_full and network in file_part) and not is_patchy

        is_stimulus = stimulus in split_full[-6] and stimulus in split_part[-9]
        if is_attribute and is_network and is_stimulus:
            path_full = path_full + "/"
            path_part = path_part + "/"
            if "variance" in file_full and "variance" in file_part:
                var_full_f = open(curr_dir + path_full + file_full, "r")
                var_part_f = open(curr_dir + path_part + file_part, "r")
                variance_full_l.append(float(var_full_f.read()))
                variance_part_l.append(float(var_part_f.read()))
                var_full_f.close()
                var_part_f.close()
            if "mean" in file_full and "mean" in file_part:
                label = file_full\
                    .replace(network, "", 1)\
                    .replace(stimulus, "")\
                    .replace("function", "")\
                    .replace("tuning", "")\
                    .replace("img_prop", "")\
                    .replace("mean", "")\
                    .replace("error", "")\
                    .replace("0.8", "")\
                    .replace(".txt", "")\
                    .replace("_", " ")
                for a in attributes:
                    label = label.replace(a, "")

                label_names.append(label)
                mean_full_f = open(curr_dir + path_full + file_full, "r")
                mean_part_f = open(curr_dir + path_part + file_part, "r")
                mean_full_l.append(float(mean_full_f.read()))
                mean_part_l.append(float(mean_part_f.read()))
                mean_full_f.close()
                mean_part_f.close()
                if mean_full_l[-1] < 0 or mean_part_l[-1] < 0:
                    raise ValueError("Negative Error is not possible")
                inf_loss_l.append(information_loss(mean_full_l[-1], mean_part_l[-1]))

    if plot_type == PLOT_TYPE["bar"]:
        x_axis = np.arange(0, len(label_names))
        plt.figure(figsize=(10, 8))
        plt.bar(x_axis + 0.0, mean_full_l, label="Error Full", width=0.25)
        plt.bar(x_axis + 0.25, mean_part_l, label="Error Part", width=0.25)
        plt.bar(x_axis + 0.5, inf_loss_l, label="Lost Information", width=0.25)
        plt.ylim(-.5, 1.)
        plt.xticks(ticks=x_axis, labels=label_names, rotation=80)
        plt.title("Lost information with sparse sampling of 80%")
        plt.ylabel("Error / Lost information")
        plt.legend()
        plt.subplots_adjust(bottom=plt_bottom_margin)
        if not save_plot:
            plt.show()
        else:
            curr_dir = os.getcwd()
            save_name = curr_dir + "/figures/lost-information/%s_network_%s_80.png" % (network, attributes[0])
            plt.savefig(save_name)
            plt.close()

    elif plot_type == PLOT_TYPE["gauss"]:
        fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
        fig.set_size_inches(10, 5)
        x_axis = np.arange(0, 1, 0.01)
        for m, v, l in zip(mean_full_l, variance_full_l, label_names):
            ax[0].plot(x_axis, norm.pdf(x_axis, m, np.sqrt(v)), label=l)
        ax[0].legend()
        ax[0].set_title("Full Sampling")
        ax[0].set_xlabel("Error")
        ax[0].set_ylabel("Probability Density")
        for m, v, l in zip(mean_part_l, variance_part_l, label_names):
            ax[1].plot(x_axis, norm.pdf(x_axis, m, np.sqrt(v)), label=l)
        ax[1].legend()
        ax[1].set_title("Sparse Sampling")
        ax[1].set_xlabel("Error")
        if not save_plot:
            plt.show()
        else:
            curr_dir = os.getcwd()
            save_name = curr_dir + "/figures/lost-information/gauss_dist_%s_network_%s_80.png" % (network, attributes[0])
            plt.savefig(save_name)
            plt.close()


if __name__ == '__main__':
    cmd_par = arg_parse_plts()

    plt_type = PLOT_TYPE["bar"]
    save_plot = True
    path_full = "experiments/error-full/error"
    path_part = "experiments/error-80-net/error"
    networks = [""]
    tunings = [""]
    stimuli = [""]
    if cmd_par.type is not None:
        plt_type = PLOT_TYPE[cmd_par.type]

    if cmd_par.show:
        save_plot = False

    if cmd_par.path_full is not None:
        path_full = cmd_par.path_full

    if cmd_par.path_part is not None:
        path_part = cmd_par.path_part

    if cmd_par.network is not None:
        if cmd_par.network.lower() == "all":
            networks = list(NETWORK_TYPE.keys())
        elif cmd_par.network.lower() in list(NETWORK_TYPE.keys()):
            networks[0] = cmd_par.network.lower()
        else:
            raise ValueError("Not a valid network type")

    if cmd_par.tuning is not None:
        if cmd_par.tuning.lower() == "all":
            tunings = list(TUNING_FUNCTION.keys())
        elif cmd_par.tuning.lower() in list(TUNING_FUNCTION.keys()):
            tunings[0] = cmd_par.tuning.lower()
        else:
            raise ValueError("Not a valid tuning function")

    if cmd_par.input is not None:
        if cmd_par.input.lower() == "all":
            stimuli = list(INPUT_TYPE.keys())
        elif cmd_par.input.lower() in list(INPUT_TYPE.keys()):
            stimuli[0] = cmd_par.input.lower()
        else:
            raise ValueError("Not a valid stimulus type")

    for network in networks:
        for tuning in tunings:
            for stimulus in stimuli:
                main_error(
                    path_full=path_full,
                    path_part=path_part,
                    network=network,
                    stimulus=stimulus,
                    attributes=[tuning],
                    plt_bottom_margin=0.3,
                    plot_type=plt_type,
                    save_plot=save_plot
                )


