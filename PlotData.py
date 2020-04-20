#!/usr/bin/python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from pathlib import Path

from modules.loadData import read_files
from modules.thesisUtils import arg_parse_plts


def get_parameters():
    cmd_par = arg_parse_plts()

    parameter_dict = {
        "path": "experiments/error",
        "x": None,
        "y": None,
        "group": None,
        "network": None,
        "stimulus": None,
        "experiment": None,
        "sampling": None,
        "parameter": None,
        "measure": None,
        "name": "Violin plot",
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

    if cmd_par.name is not None:
        parameter_dict["name"] = cmd_par.name.lower()

    return parameter_dict


def filter_dataframe(df, params, ignore_sampling=False):
    new_df = df.copy()
    save_string = params["x"] + "_" + params["y"]
    if params["network"] is not None:
        save_string += "_%s" % params["network"]
        new_df = new_df[new_df["network"] == params["network"]]
    if params["stimulus"] is not None:
        save_string += "_%s" % params["stimulus"]
        new_df = new_df[new_df["stimulus"] == params["stimulus"]]
    if params["experiment"] is not None:
        save_string += "_%s" % params["experiment"]
        new_df = new_df[new_df["experiment"] == params["experiment"]]
    if params["sampling"] is not None and not ignore_sampling:
        save_string += "_%s" % params["sampling"]
        new_df = new_df[new_df["sampling"] == params["sampling"]]
    if params["parameter"] is not None:
        save_string += "_%s" % params["parameter"]
        new_df = new_df[new_df["parameter"] == params["parameter"]]
    if params["measure"] is not None and params["measure"] != "li":
        save_string += "_%s" % params["measure"]
        new_df = new_df[new_df["measure"] == params["measure"]]

    new_df.dropna()
    return new_df, save_string


def create_li_df(df):
    df_data_full = df[np.logical_and(df["sampling"] == "1.0", df["measure"] == "distance")].sort_values(
        by=list(df.columns)
    ).drop(columns="sampling", inplace=False)
    data_full = df_data_full["value"]
    data_80 = df[np.logical_and(df["sampling"] == "0.8", df["measure"] == "distance")].sort_values(
        by=list(df.columns)
    )["value"]
    data_60 = df[np.logical_and(df["sampling"] == "0.6", df["measure"] == "distance")].sort_values(
        by=list(df.columns)
    )["value"]
    data_40 = df[np.logical_and(df["sampling"] == "0.4", df["measure"] == "distance")].sort_values(
        by=list(df.columns)
    )["value"]

    loss_80 = np.asarray(data_80) - np.asarray(data_full)
    loss_60 = np.asarray(data_60) - np.asarray(data_full)
    loss_40 = np.asarray(data_40) - np.asarray(data_full)

    df_li = df_data_full.copy()
    df_li["li_type"] = np.full(loss_80.size, "Information Loss Full:80%")
    df_li["li"] = loss_80

    df_data_full_60 = df_data_full.copy()
    df_data_full_60["li_type"] = np.full(loss_60.size, "Information Loss Full:60%")
    df_data_full_60["li"] = loss_60
    df_li = df_li.append(df_data_full_60, ignore_index=True)

    df_data_full_40 = df_data_full.copy()
    df_data_full_40["li_type"] = np.full(loss_40.size, "Information Loss Full:40%")
    df_data_full_40["li"] = loss_40
    df_li = df_li.append(df_data_full_40, ignore_index=True)

    return df_li


def violin_plot(df, params):
    new_df, save_string = filter_dataframe(df, params)
    new_df.sort_values(params["x"], inplace=True)
    figure = plt.gcf()
    figure.set_size_inches((15, 8))
    ax = figure.add_subplot(1, 2 if len(new_df[new_df.value > 1]) > 0 else 1, 1)
    sb.violinplot(
        x=params["x"],
        y=params["y"],
        hue=params["group"],
        data=new_df[new_df.value <= 1].sort_values(params["group"]),
        inner="quartile",
        palette="muted",
        ax=ax
    )
    ax.set_ylim(0., 1.)
    if len(new_df[new_df.value > 1]) > 0:
        ax.set_title("Error distribution for Error values <= 1")
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

    figure.suptitle(params["name"], fontsize=16)
    if params["save_plot"]:
        curr_dir = os.getcwd()
        Path(curr_dir + "/figures/data_analysis").mkdir(parents=True, exist_ok=True)
        save_name = curr_dir + "/figures/data_analysis/%s_violin_plot.png" % save_string
        plt.savefig(save_name)
        plt.close()
    else:
        plt.show()


def information_loss_plot(df, params):
    new_df, save_string = filter_dataframe(df, params, ignore_sampling=True)
    df_li = create_li_df(new_df)
    df_li.sort_values("li_type", inplace=True)
    plt.figure(figsize=(15, 8))
    sb.boxenplot(
        x="li_type",
        y="li",
        hue=params["x"],
        data=df_li,
        dodge=True,
    )
    plt.title(params["name"], fontsize=16)
    plt.xlabel("Type of Lost Information")
    plt.ylabel("Lost Information")
    plt.ylim(-0.5, 1.)
    if params["save_plot"]:
        curr_dir = os.getcwd()
        Path(curr_dir + "/figures/data_analysis").mkdir(parents=True, exist_ok=True)
        save_name = curr_dir + "/figures/data_analysis/%s_lost_information.png" % save_string
        plt.savefig(save_name)
        plt.close()
    else:
        plt.show()

    plt.show()


def main():
    params = get_parameters()
    df = read_files(params["path"])
    if params["measure"] == "distance":
        violin_plot(df, params)
    elif params["measure"] == "li":
        information_loss_plot(df, params)
    else:
        raise ValueError("The passed measure cannot be plotted")


if __name__ == '__main__':
    main()