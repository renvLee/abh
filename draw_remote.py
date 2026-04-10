import warnings

import pandas as pd
import numpy as np
import os
import copy


METHOD_ORDER = [
    "smt_wa",
    "smt_nw",
    "jrs_wa",
    "jrs_bwq",
    "jrs_nw_l",
    "ls",
    "jrs_mc",
    "i_ilp",
    "i_omt",
    "cg",
    "jrs_nw",
    "smt_fr",
    "cp_wa",
    "ls_tb",
    "ls_pl",
    "smt_pr",
    "dt",
    "sa",
]

marker_dict = {
    "jrs_wa": "o",
    "jrs_mc": "o",
    "jrs_nw_l": "^",
    "jrs_nw": "^",
    "ls": "s",
    "i_ilp": "s",
    "cg": "v",
    "smt_wa": "D",
    "jrs_bwq": "D",
    "cp_wa": "*",
    "smt_pr": "*",
    "i_omt": "p",
    "ls_tb": "p",
    "ls_pl": "p",
    "smt_nw": "X",
    "smt_fr": "X",
    "dt": ".",
    "sa":"H",
}

dash_dict = {name: (2, 2) for name in METHOD_ORDER}

ALPHA_REJ = 0.5

DEFAULT_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "H", "p", "8"]

_temp_morandi = ["#F0F0F0", "#E0E0E0", "#C0C0C0", "#8B8680", "#808080"]


def _normalize_name_column(df: pd.DataFrame) -> pd.DataFrame:
    df = copy.deepcopy(df)
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()
    return df


def _get_style_order(df: pd.DataFrame) -> list:
    names = [str(x).strip() for x in df["name"].dropna().unique()]
    ordered = [name for name in METHOD_ORDER if name in names]
    ordered.extend([name for name in names if name not in ordered])
    return ordered


def _get_marker_map(df: pd.DataFrame) -> dict:
    mapping = copy.deepcopy(marker_dict)
    for i, name in enumerate(_get_style_order(df)):
        if name not in mapping:
            mapping[name] = DEFAULT_MARKERS[i % len(DEFAULT_MARKERS)]
    return mapping


def get_palette(palette: int):
    import seaborn as sns

    if palette == 0:
        return sns.color_palette(
            [
                "#B0B1B6",
                "#BEB1A8",
                "#8A95A9",
                "#99857E",
                "#686789",
                "#B77F70",
                "#B57C82",
                "#9FABB9",
                "#ECCED0",
                "#91A0A5",
                "#E5E2B9",
                "#88878D",
                "#E8D3C0",
                "#7D7465",
                "#789798",
                "#7A8A71",
                "#9AA690",
                "#EA0451",
            ]
        )

    return sns.color_palette(
        [
            "#686789",
            "#B77F70",
            "#E5E2B9",
            "#BEB1A8",
            "#A79A89",
            "#8A95A9",
            "#ECCED0",
            "#7D7465",
            "#E8D3C0",
            "#7A8A71",
            "#789798",
            "#B57C82",
            "#9FABB9",
            "#B0B1B6",
            "#99857E",
            "#88878D",
            "#91A0A5",
            "#35C01F"
        ]
    )


def get_schedulability(data: pd.DataFrame, var: str, data_logs):
    data = copy.deepcopy(data)
    data[var] = data["data_id"].map(dict(zip(data_logs["id"], data_logs[var])))

    group_index = ["name", "flag", var]
    grouped_data = (
        data.groupby(group_index, as_index=False)["data_id"].count().groupby("flag")
    )

    schedulability = pd.merge(
        left=grouped_data.get_group("successful")[["name", var, "data_id"]].rename(
            columns={"data_id": "num_successful"}
        ),
        right=grouped_data.get_group("infeasible")[["name", var, "data_id"]].rename(
            columns={"data_id": "num_infeasible"}
        ),
        how="outer",
        on=["name", var],
    )
    schedulability = pd.merge(
        left=schedulability,
        right=grouped_data.get_group("unknown")[["name", var, "data_id"]].rename(
            columns={"data_id": "num_unknown"}
        ),
        how="outer",
        on=["name", var],
    )
    schedulability = schedulability.fillna(0)

    # lower bound schedulability
    schedulability["schedulability"] = (schedulability["num_successful"]) / (
        schedulability["num_successful"]
        + schedulability["num_infeasible"]
        + schedulability["num_unknown"]
    )

    return schedulability


def test_evidence_thres(stat: pd.DataFrame, var: str, confidence=0.9):
    stat_pass = stat[
        stat["num_unknown"]
        <= (stat["num_successful"] + stat["num_infeasible"] + stat["num_unknown"])
        * confidence
    ]
    stat_rej = stat[
        stat["num_unknown"]
        > (stat["num_successful"] + stat["num_infeasible"] + stat["num_unknown"])
        * confidence
    ]

    if stat[var].dtype == "int" or stat[var].dtype == "int64":
        var_range = stat[var].unique()
        var_range.sort()
        stat_rej = stat_rej.sort_values(["name", var]).reset_index(drop=True)
        addition_points = []
        for i, row in stat_rej.iterrows():
            var_index = np.where(var_range == row[var])[0][0]
            if (
                var_index - 1 >= 0
                and var_range[var_index - 1]
                not in stat_rej[stat_rej["name"] == row["name"]][var].unique()
            ):
                addition_points.append(
                    stat_pass.loc[
                        (stat_pass["name"] == row["name"])
                        & (stat_pass[var] == var_range[var_index - 1])
                    ]
                )
            if (
                var_index + 1 < len(var_range)
                and var_range[var_index + 1]
                not in stat_rej[stat_rej["name"] == row["name"]][var].unique()
            ):
                addition_points.append(
                    stat_pass.loc[
                        (stat_pass["name"] == row["name"])
                        & (stat_pass[var] == var_range[var_index + 1])
                    ]
                )
        stat_rej = pd.concat([stat_rej] + addition_points)

    stat_pass = stat_pass.fillna(0).reset_index(drop=True)
    stat_rej = stat_rej.fillna(0).reset_index(drop=True)

    return stat_pass, stat_rej


def remove_duplicate_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    unique_labels = []
    unique_handles = []
    for i in range(len(labels)):
        label = labels[i]
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handles[i])
    return unique_handles[::-1], unique_labels[::-1]


def draw_streams(df: pd.DataFrame, file_name: str, ax, data_logs):
    return draw_fig4(df, "num_stream", "Number of Streams", file_name, ax, data_logs)


def draw_bridges(df: pd.DataFrame, file_name: str, ax, data_logs):
    return draw_fig4(df, "num_sw", "Number of Bridges", file_name, ax, data_logs)


def draw_frames(df: pd.DataFrame, file_name: str, ax, path, data_logs):
    frames_list = []
    for piid in data_logs["id"]:
        task = pd.read_csv(path + '/' + str(piid) + "_task.csv")
        cycle = np.lcm.reduce(task["period"])
        frames = 0
        for period in task["period"]:
            frames += cycle / period
        frames = np.power(2, np.log2(frames).astype(int))
        frames_list.append(frames)
    data_logs["num_frame"] = frames_list
    return draw_fig4(df, "num_frame", "Number of Frames", file_name, ax, data_logs)


def draw_links(df: pd.DataFrame, file_name: str, ax, path, data_logs):
    links_list = []
    for piid in data_logs["id"]:
        topo = pd.read_csv(path + '/' + str(piid) + "_topo.csv")
        links = len(topo["link"])
        links = (links // 50 + 1) * 50  # discreet
        links_list.append(links)
    data_logs["num_link"] = links_list
    return draw_fig4(df, "num_link", "Number of Links", file_name, ax, data_logs)


def draw_fig4(
    df: pd.DataFrame, var: str, graph_name: str, file_name: str, ax, data_logs
):
    import seaborn as sns
    import matplotlib.pyplot as plt

    schedulability = _normalize_name_column(get_schedulability(df, var, data_logs))
    stat_pass, stat_rej = test_evidence_thres(schedulability, var)
    style_order = _get_style_order(schedulability)
    dynamic_markers = _get_marker_map(schedulability)
    palette = get_palette(0)

    # plot rejected points
    ax = sns.lineplot(
        ax=ax,
        data=stat_rej,
        x=var,
        y="schedulability",
        hue="name",
        style="name",
        palette=palette,
        hue_order=METHOD_ORDER,
        style_order=style_order,
        markers=dynamic_markers,
        dashes=dash_dict,
        alpha=ALPHA_REJ,
        markeredgecolor=None,
        fillstyle="none",
        linewidth=2.4,
        markersize=12,
    )

    for stat in list([x[1].reset_index(drop=True) for x in stat_pass.groupby("name")]):
        ax = sns.lineplot(
            ax=ax,
            data=stat,
            x=var,
            y="schedulability",
            hue="name",
            style="name",
            palette=palette,
            hue_order=METHOD_ORDER,
            style_order=style_order,
            markers=dynamic_markers,
            dashes=False,
            markeredgecolor=None,
            fillstyle="none",
            linewidth=2.4,
            markersize=12,
        )
    ax.grid(axis="y")
    ax.set_ylim(0, 1)
    ax.set_xlabel(graph_name, fontsize=16)
    ax.set_ylabel("Schedulable Ratio", fontsize=16)
    legend = ax.legend(
        *remove_duplicate_legend(ax),
        ncol=3,
        loc="upper center",
        prop={"size": 10},
        mode="expand",
        bbox_to_anchor=(0.0, 1.4, 1.0, 0),
        frameon=False,
    )
    legend.remove()
    return ax


def draw_period(df: pd.DataFrame, file_name: str, ax, data_logs):
    period_dict = {3: "Harmonic Sparse", 4: "Harmonic Dense"}
    data_logs["period"] = data_logs["period"].apply(lambda x: period_dict[x])
    schedulability = get_schedulability(df, "period", data_logs)
    draw_fig5(schedulability, "period", list(period_dict.values()), file_name, ax)


def draw_payload(df: pd.DataFrame, file_name: str, ax, data_logs):
    size_dict = {2: "Small"}
    data_logs["size"] = data_logs["size"].apply(lambda x: size_dict[x])
    schedulability = get_schedulability(df, "size", data_logs)
    draw_fig5(schedulability, "size", list(size_dict.values()), file_name, ax)


def draw_deadline(df: pd.DataFrame, file_name: str, ax, data_logs):
    deadline_dict = {1: "Implicit"}
    data_logs["deadline"] = data_logs["deadline"].apply(lambda x: deadline_dict[x])
    schedulability = get_schedulability(df, "deadline", data_logs)
    draw_fig5(schedulability, "deadline", list(deadline_dict.values()), file_name, ax)


def draw_topo(df: pd.DataFrame, file_name: str, ax, data_logs):
    topo_dict = {0: "Line", 1: "Ring", 2: "Tree", 3: "Mesh"}
    data_logs["topo"] = data_logs["topo"].apply(lambda x: topo_dict[x])
    schedulability = get_schedulability(df, "topo", data_logs)
    draw_fig5(schedulability, "topo", ["Line", "Ring"], file_name, ax)


def draw_fig5(df: pd.DataFrame, var: str, hue_order: list, file_name: str, ax):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.rc("xtick", labelsize=12)
    plt.rcParams["axes.axisbelow"] = True
    ax = sns.barplot(
        ax=ax,
        data=df,
        y="schedulability",
        x="name",
        hue=var,
        hue_order=hue_order,
        palette=get_palette(1),
        order=METHOD_ORDER,
    )
    ax.set_xlabel("")
    ax.grid(axis="y")
    ax.set_yticks(np.arange(0, 1.00001, step=0.2))
    ax.set_ylabel("Schedulable Ratio")
    legend = ax.legend(
        *remove_duplicate_legend(ax),
        ncol=6,
        loc="upper center",
        prop={"size": 10},
        mode="expand",
        bbox_to_anchor=(0.0, 1.4, 1.0, 0),
        frameon=False,
    )
    legend.remove()


def get_comparison_matrix(df: pd.DataFrame):
    index_map = {method: i for i, method in enumerate(df["name"].unique())}
    num_methods = len(index_map)
    group_index = ["data_id"]

    single_data = df[["name", "data_id", "flag"]]
    paired_data = pd.merge(
        left=single_data[single_data["flag"] != "unknown"],
        right=single_data[single_data["flag"] != "unknown"],
        on=group_index,
    ).dropna()

    comparison_matrix = np.zeros((num_methods, num_methods))
    all_result_matrix = np.zeros((num_methods, num_methods))

    for i, row in paired_data.iterrows():
        x = row["name_x"]
        y = row["name_y"]
        if (row["flag_x"] == "successful") and (row["flag_y"] == "infeasible"):
            comparison_matrix[index_map[x], index_map[y]] += 1
        all_result_matrix[index_map[x], index_map[y]] += 1

    return comparison_matrix, all_result_matrix


def draw_comparison_matrix(df: pd.DataFrame, file_name: str, ax):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    default_color = "#FFEBCD"
    morandi_cmap = mcolors.LinearSegmentedColormap.from_list(
        "morandi_cmap", _temp_morandi
    )
    morandi_cmap.set_bad(color=default_color)
    extended_colors = [default_color] + _temp_morandi
    extended_cmap = mcolors.LinearSegmentedColormap.from_list(
        "morandi_cmap", extended_colors
    )
    comparison_matrix, all_result_matrix = get_comparison_matrix(df)
    comparison_matrix[np.where(comparison_matrix == 0)] = np.nan

    methods = df["name"].unique()
    num_methods = len(methods)
    dominate_matrix = np.empty([num_methods, num_methods], dtype=str)

    for i in range(num_methods):
        for j in range(num_methods):
            sa_ij = comparison_matrix[i][j]
            sa_ji = comparison_matrix[j][i]
            if sa_ij > 0 and np.isnan(sa_ji):
                dominate_matrix[i][j] = "✗"

    means = np.nanmean(np.nan_to_num(comparison_matrix / all_result_matrix), axis=0)
    sorted_indices = np.argsort(means)
    sorted_comparison_matrix = (comparison_matrix / all_result_matrix)[
        :, sorted_indices
    ][sorted_indices, :]
    sorted_dominate_matrix = dominate_matrix[:, sorted_indices][sorted_indices, :]

    sns.heatmap(
        ax=ax,
        data=sorted_comparison_matrix,
        xticklabels=[methods[x] for x in sorted_indices],
        yticklabels=[methods[x] for x in sorted_indices],
        cmap=extended_cmap,
        cbar_kws={"label": "Schedulability Advantage"},
        linewidths=1,
        linecolor="white",
        vmin=0,
    )
    sns.heatmap(
        ax=ax,
        data=sorted_comparison_matrix,
        xticklabels=[methods[x] for x in sorted_indices],
        yticklabels=[methods[x] for x in sorted_indices],
        cmap=morandi_cmap,
        cbar=False,
        linewidths=1,
        linecolor="white",
        vmin=0,
        annot=sorted_dominate_matrix,
        fmt="",
    )

    plt.xticks(rotation=45, ha="right")


def get_runtime_stat(data: pd.DataFrame, var: str):
    data = data[(data["flag"] != "unknown") | (data["total_mem"] < 4000)]
    data.loc[:, ["total_time"]] = data["total_time"] / 60
    return data.groupby([var, "name"], as_index=False)["total_time"].mean()


def get_memory_stat(data: pd.DataFrame, var: str):
    data = data[(data["flag"] != "unknown") | (data["total_time"] < 7200)]
    return data.groupby([var, "name"], as_index=False)["total_mem"].mean()


def draw_scalability(
    df: pd.DataFrame,
    x: str,
    y: str,
    x_label: str,
    y_label: str,
    file_name: str,
    ax,
    data_logs,
):
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = _normalize_name_column(df)
    df[x] = df["data_id"].map(dict(zip(data_logs["id"], data_logs[x])))

    plt.rcParams["axes.axisbelow"] = True

    stat = get_runtime_stat(df, x) if y == "total_time" else get_memory_stat(df, x)

    schedulability = get_schedulability(df, x, data_logs)
    pass_rej = test_evidence_thres(schedulability, x)
    stat_pass = pd.merge(stat, pass_rej[0], on=["name", x])
    stat_rej = pd.merge(stat, pass_rej[1], on=["name", x])
    style_df = pd.concat([stat_pass[["name"]], stat_rej[["name"]]], ignore_index=True)
    style_order = _get_style_order(style_df)
    dynamic_markers = _get_marker_map(style_df)
    palette = get_palette(0)

    ax = sns.lineplot(
        ax=ax,
        data=stat_pass,
        x=x,
        y=y,
        hue="name",
        style="name",
        palette=palette,
        hue_order=METHOD_ORDER,
        style_order=style_order,
        markers=dynamic_markers,
        dashes=False,
        markeredgecolor=None,
        fillstyle="none",
        linewidth=2.4,
        markersize=12,
    )

    ax = sns.lineplot(
        ax=ax,
        data=stat_rej,
        x=x,
        y=y,
        hue="name",
        style="name",
        palette=palette,
        hue_order=METHOD_ORDER,
        style_order=style_order,
        markers=dynamic_markers,
        dashes=dash_dict,
        alpha=ALPHA_REJ,
        markeredgecolor=None,
        fillstyle="none",
        linewidth=2.4,
        markersize=12,
    )

    ax.grid(axis="y")
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    legend = ax.legend(
        *remove_duplicate_legend(ax),
        ncol=3,
        loc="upper center",
        prop={"size": 10},
        mode="expand",
        bbox_to_anchor=(0.0, 1.4, 1.0, 0),
        frameon=False,
    )
    legend.remove()


def draw_runtime(df: pd.DataFrame, file_name: str, ax1, ax2, data_logs):
    draw_scalability(
        df,
        "num_stream",
        "total_time",
        "Number of streams",
        "Runtime (Mins)",
        f"{file_name}_stream",
        ax1,
        data_logs,
    )
    draw_scalability(
        df,
        "num_sw",
        "total_time",
        "Number of bridges",
        "Runtime (Mins)",
        f"{file_name}_bridge",
        ax2,
        data_logs,
    )


def draw_mem(df: pd.DataFrame, file_name: str, ax1, ax2, data_logs):
    draw_scalability(
        df,
        "num_stream",
        "total_mem",
        "Number of streams",
        "Memory (MB)",
        f"{file_name}_stream",
        ax1,
        data_logs,
    )
    draw_scalability(
        df,
        "num_sw",
        "total_mem",
        "Number of bridges",
        "Memory (MB)",
        f"{file_name}_bridge",
        ax2,
        data_logs,
    )


def draw_legend():
    import matplotlib.pyplot as plt

    legend_fig = plt.figure(figsize=(10, 2))
    handles = [
        plt.Line2D(
            [0],
            [0],
            color="none",
            marker=marker_dict[METHOD_ORDER[i]],
            linestyle="",
            markersize=7,
            markeredgecolor=get_palette(0)[i],
        )
        for i in range(len(METHOD_ORDER))
    ]
    legend_fig.legend(
        handles,
        METHOD_ORDER,
        loc="center",
        ncol=9,
        prop={"size": 8},
        numpoints=1,
        handletextpad=0,
    )
    legend_fig.savefig("legend.pdf", bbox_inches="tight")


def draw(csv_path: str, data_path: str, output_affix="./"):
    warnings.filterwarnings("ignore")
    df = _normalize_name_column(pd.read_csv(csv_path))
    data_logs = pd.read_csv(os.path.join(data_path, "dataset_logs.csv"))

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(36, 12))
    gs = gridspec.GridSpec(6, 5, figure=fig)

    plt.rcParams["axes.axisbelow"] = True

    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[:2, 1])
    ax3 = fig.add_subplot(gs[2:4, 0])
    ax4 = fig.add_subplot(gs[2:4, 1])

    draw_streams(df, f"{output_affix}stream", ax1, data_logs)
    draw_bridges(df, f"{output_affix}bridge", ax2, data_logs)
    draw_links(df, f"{output_affix}link", ax3, data_path, data_logs)
    draw_frames(df, f"{output_affix}frame", ax4, data_path, data_logs)

    ax5 = fig.add_subplot(gs[0, 2:])
    ax6 = fig.add_subplot(gs[1, 2:])
    ax7 = fig.add_subplot(gs[2, 2:])
    ax8 = fig.add_subplot(gs[3, 2:])

    draw_topo(df, f"{output_affix}topo", ax5, data_logs)
    draw_period(df, f"{output_affix}period", ax6, data_logs)
    draw_payload(df, f"{output_affix}payload", ax7, data_logs)
    draw_deadline(df, f"{output_affix}deadline", ax8, data_logs)

    ax9 = fig.add_subplot(gs[4:, 4])

    draw_comparison_matrix(df, f"{output_affix}comparison_matrix", ax9)

    ax10 = fig.add_subplot(gs[4:, 0])
    ax11 = fig.add_subplot(gs[4:, 1])
    ax12 = fig.add_subplot(gs[4:, 2])
    ax13 = fig.add_subplot(gs[4:, 3])

    draw_runtime(df, f"{output_affix}runtime", ax10, ax11, data_logs)
    draw_mem(df, f"{output_affix}mem", ax12, ax13, data_logs)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color="none",
            marker=marker_dict[METHOD_ORDER[i]],
            linestyle="",
            markersize=12,
            markeredgecolor=get_palette(0)[i],
        )
        for i in range(len(METHOD_ORDER))
    ]
    fig.legend(
        handles,
        METHOD_ORDER,
        loc="upper center",
        ncol=17,
        prop={"size": 16},
        numpoints=1,
        handletextpad=0,
        bbox_to_anchor=(0.5, 1.05),
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw("./results.csv", ".")
