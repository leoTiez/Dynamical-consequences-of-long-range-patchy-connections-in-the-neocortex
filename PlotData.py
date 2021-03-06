#!/usr/bin/python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from pathlib import Path

from modules.loadData import read_files
from modules.thesisUtils import arg_parse_plts


def get_parameters():
    """
    Function to get command line parameters and set the parameters used in this script accordingly
    :return: Dictionary with parameters
    """
    cmd_par = arg_parse_plts(sys.argv[1:])

    parameter_dict = {
        "path": "experiments/new/error",
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
        parameter_dict["x"] = cmd_par.x

    if cmd_par.y is not None:
        parameter_dict["y"] = cmd_par.y

    if cmd_par.group is not None:
        parameter_dict["group"] = cmd_par.group

    if cmd_par.network is not None:
        parameter_dict["network"] = cmd_par.network

    if cmd_par.input is not None:
        parameter_dict["stimulus"] = cmd_par.input

    if cmd_par.experiment is not None:
        parameter_dict["experiment"] = cmd_par.experiment

    if cmd_par.sampling is not None:
        parameter_dict["sampling"] = cmd_par.sampling

    if cmd_par.parameter is not None:
        parameter_dict["parameter"] = cmd_par.parameter

    if cmd_par.measure is not None:
        parameter_dict["measure"] = cmd_par.measure

    if cmd_par.title is not None:
        parameter_dict["name"] = cmd_par.title

    return parameter_dict


def filter_dataframe(df, params, ignore_sampling=False):
    """
    Filter dataframe according to parameters
    :param df: Dataframe
    :param params: Parameters
    :param ignore_sampling: When set to true, ignore the sampling rate if set as parameter. This is important when
    information loss is plotted
    :return: New filtered dataframe
    """
    new_df = df.copy()
    save_string = params["x"] + "_" + params["y"]
    if params["network"] is not None:
        save_string += "_network_%s" % params["network"]
        new_df = new_df[new_df["network"] == params["network"].replace("\\n", "\n")]
    if params["stimulus"] is not None:
        save_string += "_input_%s" % params["stimulus"]
        new_df = new_df[new_df["stimulus"] == int(params["stimulus"])]
    if params["experiment"] is not None:
        save_string += "_experiment_%s" % params["experiment"]
        new_df = new_df[new_df["experiment"] == params["experiment"]]
    if params["sampling"] is not None and not ignore_sampling:
        save_string += "_sampling_%s" % params["sampling"]
        new_df = new_df[new_df["sampling"] == params["sampling"]]
    if params["parameter"] is not None:
        save_string += "_parameter_%s" % params["parameter"]
        new_df = new_df[new_df["parameter"] == params["parameter"]]
    if params["measure"] is not None and params["measure"] != "li":
        save_string += "_measure_%s" % params["measure"]
        new_df = new_df[new_df["measure"] == params["measure"]]

    new_df.dropna()
    return new_df, save_string


def create_li_df(df):
    """
    Create new dataframe that is used for determining the information loss
    :param df: The dataframe with the loaded data
    :return: New datafram that contains the lost information for different sampling rates
    """
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
    """
    Plot violin plot
    :param df: Dataframe
    :param params: Parameters
    :return: None
    """
    plt.rcParams.update({"font.size": 20})
    new_df, save_string = filter_dataframe(df, params)
    figure = plt.gcf()
    figure.set_size_inches((15, 10))
    ax = figure.add_subplot(1, 2 if len(new_df[new_df.value > 1]) > 0 else 1, 1)

    sb.boxplot(
        x=params["x"],
        y=params["y"],
        hue=params["group"],
        data=new_df[new_df.value <= 1].sort_values(params["group"]),
        dodge=True,
        order=sorted(new_df[params["x"]].drop_duplicates().tolist()),
        ax=ax,
        boxprops={"alpha":0.2}
    )

    sb.swarmplot(
        x=params["x"],
        y=params["y"],
        hue=params["group"],
        data=new_df[new_df.value <= 1].sort_values(params["group"]),
        order=sorted(new_df[params["x"]].drop_duplicates().tolist()),
        ax=ax,
        dodge=True,
    )

    ax.set_ylim(0., 1.)
    handles, lables = ax.get_legend_handles_labels()
    num_labels = len(new_df[params["group"]].drop_duplicates())
    ax.legend(handles[:num_labels], lables[:num_labels])

    if len(new_df[new_df.value > 1]) > 0:
        ax.set_title("Error distribution for Error values <= 1")
        ax_2 = figure.add_subplot(1, 2, 2)
        sb.boxplot(
            x=params["x"],
            y=params["y"],
            hue=params["group"],
            data=new_df[new_df.value > 1].sort_values(params["group"]),
            order=sorted(new_df[params["x"]].drop_duplicates().tolist()),
            dodge=True,
            boxprops={"alpha": 0.2},
            ax=ax_2
        )
        sb.swarmplot(
            x=params["x"],
            y=params["y"],
            hue=params["group"],
            data=new_df[new_df.value > 1].sort_values(params["group"]),
            order=sorted(new_df[params["x"]].drop_duplicates().tolist()),
            ax=ax,
            dodge=True,
        )
        ax_2.set_title("Error distribution for Error values > 1")

    figure.suptitle(params["name"], fontsize=20)
    ax.set_ylabel("Reconstruction Error E")
    if params["save_plot"]:
        curr_dir = os.getcwd()
        Path(curr_dir + "/figures/data_analysis").mkdir(parents=True, exist_ok=True)
        save_name = curr_dir + "/figures/data_analysis/%s_violin_plot.png" % save_string
        plt.savefig(save_name)
        plt.close()
    else:
        plt.show()


def information_loss_plot(df, params):
    """
    Plot information loss as boxen plot
    :param df: Dataframe
    :param params: Parameters
    :return: None
    """
    plt.rcParams.update({"font.size": 20})
    new_df, save_string = filter_dataframe(df, params, ignore_sampling=True)
    df_li = create_li_df(new_df)
    df_li.sort_values("li_type", inplace=True)
    plt.figure(figsize=(15, 8))
    sb.boxenplot(
        x="li_type",
        y="li",
        hue=params["x"],
        hue_order=sorted(new_df[params["x"]].drop_duplicates().tolist()),
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
    """
    Main function
    :return: None
    """
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

