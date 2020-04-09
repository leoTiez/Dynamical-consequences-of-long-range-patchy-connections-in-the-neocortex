#!/usr/bin/python3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from createThesisNetwork import NETWORK_TYPE
from modules.networkConstruction import TUNING_FUNCTION
from modules.thesisUtils import arg_parse_plts
from modules.createStimulus import INPUT_TYPE

PLOT_TYPE = {
    "bar": 0,
    "gauss": 1,
}


def check_naming(file_name, file_split, attributes, network, stimulus):
    is_attribute = np.all(np.asarray([a in file_name for a in attributes]))
    is_patchy = "patchy" in file_name
    if "patchy" in network:
        is_network = network in file_name and is_patchy
    elif "random" in network:
        is_network = network in file_split
    elif len(network) == 0:
        is_network = True
    else:
        is_network = network in file_name and not is_patchy
    # use "in" to allow empty strings
    is_stimulus = stimulus in file_split[-9]
    return is_attribute, is_network, is_stimulus


def information_loss(error_full, error_part):
    return error_part - error_full


def main_error(
        path_full,
        paths_comparison,
        network="",
        stimulus="",
        attributes=[""],
        plt_bottom_margin=0.2,
        plot_type=PLOT_TYPE["bar"],
        save_plot=False
):
    curr_dir = os.getcwd() + "/"
    variance_full_l = []
    variance_comparison = [[] for _ in range(len(paths_comparison))]

    mean_full_l = []
    mean_comparison = [[] for _ in range(len(paths_comparison))]

    total_num_comparison = sum(range(len(paths_comparison) + 1))
    inf_loss_l = [[] for _ in range(total_num_comparison)]

    label_names = []
    label_names_inf = []

    part_files = [os.listdir(curr_dir + path_part) for path_part in paths_comparison]
    for file_full in sorted(os.listdir(curr_dir + path_full)):
        is_mean_error = "mean_error.txt" in file_full
        cut_off_idx = -19 if is_mean_error else -23
        split_full = file_full.split("_")

        is_attribute, is_network, is_stimulus = check_naming(file_full, split_full, attributes, network, stimulus)
        if not (is_attribute and is_network and is_stimulus):
            continue

        files_part = []
        fail = False
        for file_p in part_files:
            files_part.append(list(filter(lambda f: file_full[:cut_off_idx] in f, file_p)))
            files_part[-1] = list(
                filter(lambda f: "mean_error.txt" in f if is_mean_error else "error_variance.txt" in f, files_part[-1])
            )
            if network == "random":
                files_part[-1] = list(
                    filter(
                        lambda f: network == f.split("_")[0] if is_mean_error else "error_variance.txt" in f.split("_")[
                            0],
                        files_part[-1]
                    )
                )
                if len(files_part[-1]) == 0:
                    fail = True
                    break

            files_part[-1] = files_part[-1][0]
            split_part = files_part[-1].split("_")
            is_attribute, is_network, is_stimulus = check_naming(
                files_part[-1],
                split_part,
                attributes,
                network,
                stimulus
            )

            if not (is_attribute and is_network and is_stimulus):
                fail = True
                break
        if fail:
            continue

        path_full = path_full + "/"
        if "variance" in file_full:
            var_full_f = open(curr_dir + path_full + file_full, "r")
            variance_full_l.append(float(var_full_f.read()))
            var_full_f.close()
            for p_part, f_part, var in zip(paths_comparison, files_part, variance_comparison):
                var_f_part = open(curr_dir + p_part + "/" + f_part, "r")
                var.append(float(var_f_part.read()))
                var_f_part.close()

        if "mean" in file_full:
            label = file_full \
                .replace(network, "", 1) \
                .replace(stimulus, "") \
                .replace("function", "") \
                .replace("tuning", "") \
                .replace("img_prop", "") \
                .replace("mean", "") \
                .replace("error", "") \
                .replace("1.0", "") \
                .replace(".txt", "")

            for a in attributes:
                label = label.replace(a, "")

            label = label.replace("__", "").replace("_", " ")
            label_names.append(label)

            mean_full_f = open(curr_dir + path_full + file_full, "r")
            mean_full_l.append(float(mean_full_f.read()))
            mean_full_f.close()
            if mean_full_l[-1] < 0:
                raise ValueError("Negative Error is not possible")
            for p_part, f_part, mean in zip(paths_comparison, files_part, mean_comparison):
                mean_f_part = open(curr_dir + p_part + "/" + f_part, "r")
                mean.append(float(mean_f_part.read()))
                mean_f_part.close()

                if mean[-1] < 0:
                    raise ValueError("Negative Error is not possible")

            counter = 0
            for num, mean in enumerate(mean_comparison):
                inf_loss_l[counter].append(information_loss(mean_full_l[-1], mean[-1]))
                counter += 1
                for mean_comp in mean_comparison[num + 1:]:
                    inf_loss_l[counter].append(information_loss(mean[-1], mean_comp[-1]))
                    counter += 1

    for num in range(len(mean_comparison)):
        label_names_inf.append("Full:%s percent" % (100 - 20 * (num + 1)))
        for num_inner in range(num + 1, len(mean_comparison)):
            label_names_inf.append("%s:%s percent" % (100 - 20 * (num + 1), 100 - 20 * (num_inner + 1)))

    if plot_type == PLOT_TYPE["bar"]:
        x_axis = np.arange(0, len(label_names))
        fig = plt.gcf()
        fig.set_size_inches(15, 8)
        ax_1 = fig.add_subplot(1, 2 if len(paths_comparison) > 0 else 1, 1)

        spacing = np.linspace(0, 1, len(mean_comparison) + 3)[1:-1]
        ax_1.bar(x_axis + 0.0, mean_full_l, label="Error Full", width=0.25)
        for num, mean, s in zip(range(len(mean_comparison)), mean_comparison, spacing):
            ax_1.bar(x_axis + s, mean, label="Error %s percent" % (100 - 20 * (num + 1)), width=spacing[0])

        ax_1.set_ylim(0., 1.)
        ax_1.set_xticks(ticks=x_axis) # + spacing[0])
        ax_1.set_xticklabels(labels=label_names, rotation=80)
        ax_1.set_title("Reconstruction Error with sparse sampling.")
        ax_1.set_ylabel("Error")
        ax_1.legend()

        if len(paths_comparison) > 0:
            ax_2 = fig.add_subplot(1, 2, 2)
            spacing = np.linspace(0, 1, len(inf_loss_l) + 3)[:-1]
            for label, il, s in zip(label_names_inf, inf_loss_l, spacing):
                ax_2.bar(x_axis + s, il, label="Lost Information %s" % label, width=spacing[1])
                ax_2.set_xticks(ticks=x_axis + spacing[1])

            ax_2.set_xticklabels(labels=label_names, rotation=80)
            ax_2.set_ylim(-.5, 1.)
            ax_2.set_title("Lost information with sparse sampling.")
            ax_2.set_ylabel("Lost information")
            ax_2.legend()

        plt.subplots_adjust(bottom=plt_bottom_margin)
        if not save_plot:
            plt.show()
        else:
            curr_dir = os.getcwd()
            save_name = curr_dir + "/figures/lost-information/%s_network_%s.png" % (network, attributes[0])
            plt.savefig(save_name)
            plt.close()

    # elif plot_type == PLOT_TYPE["gauss"]:
    #     fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
    #     fig.set_size_inches(10, 5)
    #     x_axis = np.arange(0, 1, 0.01)
    #     for m, v, l in zip(mean_full_l, variance_full_l, label_names):
    #         ax[0].plot(x_axis, norm.pdf(x_axis, m, np.sqrt(v)), label=l)
    #     ax[0].legend()
    #     ax[0].set_title("Full Sampling")
    #     ax[0].set_xlabel("Error")
    #     ax[0].set_ylabel("Probability Density")
    #     for m, v, l in zip(mean_part_l, variance_part_l, label_names):
    #         ax[1].plot(x_axis, norm.pdf(x_axis, m, np.sqrt(v)), label=l)
    #     ax[1].legend()
    #     ax[1].set_title("Sparse Sampling")
    #     ax[1].set_xlabel("Error")
    #     if not save_plot:
    #         plt.show()
    #     else:
    #         curr_dir = os.getcwd()
    #         save_name = curr_dir + "/figures/lost-information/gauss_dist_%s_network_%s_80.png" % (network, attributes[0])
    #         plt.savefig(save_name)
    #         plt.close()


if __name__ == '__main__':
    cmd_par = arg_parse_plts()

    plt_type = PLOT_TYPE["bar"]
    save_plot = True
    # path_full = "experiments/error-full/error"
    # path_comparison = ["experiments/error-80-net/error", "experiments/error-60-net/error", "experiments/error-40-net/error"]
    path_full = "experiments/patchy-exp-full/error"
    path_comparison = ["experiments/patchy-exp-80/error"]
    networks = [""]
    tunings = [""]
    stimuli = [""]
    attributes = [""]
    if cmd_par.type is not None:
        plt_type = PLOT_TYPE[cmd_par.type]

    if cmd_par.show:
        save_plot = False

    if cmd_par.path_full is not None:
        path_full = cmd_par.path_full

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

    if cmd_par.attributes is not None:
        attributes = list(map(str, cmd_par.attributes.strip('[]').split(',')))

    for network in networks:
        for tuning in tunings:
            for stimulus in stimuli:
                att = attributes.copy()
                att.append(tuning)
                main_error(
                    path_full=path_full,
                    paths_comparison=path_comparison,
                    network=network,
                    stimulus=stimulus,
                    attributes=att,
                    plt_bottom_margin=0.3,
                    plot_type=plt_type,
                    save_plot=save_plot
                )


