#!/usr/bin/python3
import os
import matplotlib.pyplot as plt
import seaborn as sb
from pathlib import Path

from modules.loadData import read_files
from modules.thesisUtils import arg_parse_plts


def get_parameters():
    cmd_par = arg_parse_plts()

    parameter_dict = {
        "path": "experiments/tuning/error",
        "x": None,
        "y": None,
        "group": None,
        "network": None,
        "stimulus": None,
        "experiment": None,
        "sampling": None,
        "parameter": None,
        "measure": None,
        "save_plot": True,
    }

    if cmd_par.show:
        parameter_dict["save_plot"] = False

    if cmd_par.path is not None:
        parameter_dict["path"] = cmd_par.path

    if cmd_par.x is not None:
        parameter_dict["x"] = cmd_par.x.lower()

    if cmd_par.y is not None:
        parameter_dict["y"] = cmd_par.y.lower()

    if cmd_par.group is not None:
        parameter_dict["group"] = cmd_par.group.lower()

    if cmd_par.network is not None:
        parameter_dict["network"] = cmd_par.network.lower()

    if cmd_par.input is not None:
        parameter_dict["stimulus"] = cmd_par.input.lower()

    if cmd_par.experiment is not None:
        parameter_dict["experiment"] = cmd_par.experiment.lower()

    if cmd_par.sampling is not None:
        parameter_dict["sampling"] = cmd_par.sampling.lower()

    if cmd_par.parameter is not None:
        parameter_dict["parameter"] = cmd_par.parameter.lower()

    if cmd_par.measure is not None:
        parameter_dict["measure"] = cmd_par.measure.lower()

    return parameter_dict


def violin_plot(df, params):
    new_df = df.copy()
    save_string = params["x"] + params["y"]
    if params["network"] is not None:
        save_string += "_%s" % params["network"]
        new_df = new_df[new_df["network"] == params["network"]]
    if params["stimulus"] is not None:
        save_string += "_%s" % params["stimulus"]
        new_df = new_df[new_df["stimulus"] == params["stimulus"]]
    if params["experiment"] is not None:
        save_string += "_%s" % params["experiment"]
        new_df = new_df[new_df["experiment"] == params["experiment"]]
    if params["sampling"] is not None:
        save_string += "_%s" % params["sampling"]
        new_df = new_df[new_df["sampling"] == params["sampling"]]
    if params["parameter"] is not None:
        save_string += "_%s" % params["parameter"]
        new_df = new_df[new_df["parameter"] == params["parameter"]]
    if params["measure"] is not None:
        save_string += "_%s" % params["measure"]
        new_df = new_df[new_df["measure"] == params["measure"]]

    new_df.dropna()

    figure = plt.gcf()
    figure.set_size_inches((15, 8))
    ax = figure.add_subplot(1, 2 if len(new_df.value > 1) > 0 else 1, 1)
    sb.violinplot(
        x=params["x"],
        y=params["y"],
        hue=params["group"],
        data=new_df[new_df.value <= 1].sort_values(params["group"]),
        inner="quartile",
        palette="muted",
        ax=ax
    )
    ax.set_title("Error distribution for Error values <= 1")
    if len(new_df.value > 1) > 0:
        ax_2 = figure.add_subplot(1, 2, 2)
        sb.violinplot(
            x=params["x"],
            y=params["y"],
            hue=params["group"],
            data=new_df[new_df.value > 1].sort_values(params["group"]),
            inner="quartile",
            palette="muted",
            ax=ax_2
        )
        ax_2.set_title("Error distribution for Error values > 1")

    if params["save_plot"]:
        curr_dir = os.getcwd()
        Path(curr_dir + "/figures/data_analysis").mkdir(parents=True, exist_ok=True)
        save_name = curr_dir + "/figures/data_analysis/%s_violin_plot.png" % save_string
        plt.savefig(save_name)
        plt.close()
    else:
        plt.show()


def main():
    params = get_parameters()
    df = read_files(params["path"])
    violin_plot(df, params)


if __name__ == '__main__':
    main()