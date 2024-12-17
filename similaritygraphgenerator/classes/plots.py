import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Plots:
    def __init__(self, name):
        self.update_name(name)

    def generate_regex(self, name):
        pattern_str = re.escape(name)
        pattern_str = pattern_str.replace(r"\?", r"\d+")
        pattern_str = f".*{pattern_str}"
        pattern_str = f"{pattern_str}.csv"
        return re.compile(f"^{pattern_str}$")

    def update_name(self, name):
        self.name = name
        self.name_pattern = self.generate_regex(name)

    def plot(
        self,
        base_folder,
        plot_type,
        x_axis,
        y_axis,
        data_point_values=None,
        algorithm=None,
        save=False,
    ):
        if plot_type == "line":
            self.plot_line_chart(base_folder, x_axis, y_axis, save)
            pass
        if plot_type == "heatmap":
            self.plot_heatmap(
                base_folder, x_axis, y_axis, data_point_values, algorithm, save
            )
            pass

    def get_heatmap_data(
        self, base_folder, x_axis, y_axis, data_points, algorithm
    ):
        x_values = []
        y_values = []
        data_point_values = []

        for root, dirs, files in os.walk(base_folder):
            x = None
            y = None
            data = None

            for file in files:
                if self.name_pattern.match(file):
                    csv_path = os.path.join(root, file)
                    df = pd.read_csv(csv_path)
                    x = df[f"{x_axis}"][0]
                    y = df[f"{y_axis}"][0]
                    data_row = df[df["Name"] == f"{algorithm}"]
                    if not data_row.empty:
                        data = data_row[f"{data_points}"].values[0]

            if x is not None and y is not None and data is not None:
                x_values.append(x)
                y_values.append(y)
                data_point_values.append(data)

        data_frame = pd.DataFrame(
            {
                f"{x_axis}": x_values,
                f"{y_axis}": y_values,
                f"{data_points}": data_point_values,
            }
        )

        heatmap_data = data_frame.pivot(
            index=f"{y_axis}", columns=f"{x_axis}", values=f"{data_points}"
        )
        heatmap_data = heatmap_data.fillna(0)

        return heatmap_data

    def get_line_chart_data(self, base_folder, x_axis, y_axis):
        x_values = []
        data_gn = []
        data_louvain = []
        data_lpa = []
        data_infomap = []
        data_gm = []

        for root, dirs, files in os.walk(base_folder):
            x = None
            y_gn = None
            y_louvain = None
            y_lpa = None
            y_infomap = None
            y_gm = None

            for file in files:
                if self.name_pattern.match(file):
                    csv_path = os.path.join(root, file)
                    df = pd.read_csv(csv_path)

                    x = df[f"{x_axis}"][0]
                    gn_row = df[df["Name"] == "girvan_newman"]
                    louvain_row = df[df["Name"] == "louvain"]
                    lpa_row = df[df["Name"] == "lpa"]
                    infomap_row = df[df["Name"] == "infomap"]
                    gm_row = df[df["Name"] == "greedy_modularity"]

                    if not gn_row.empty:
                        y_gn = gn_row[f"{y_axis}"].values[0]
                    if not louvain_row.empty:
                        y_louvain = louvain_row[f"{y_axis}"].values[0]
                    if not lpa_row.empty:
                        y_lpa = lpa_row[f"{y_axis}"].values[0]
                    if not infomap_row.empty:
                        y_infomap = infomap_row[f"{y_axis}"].values[0]
                    if not gm_row.empty:
                        y_gm = gm_row[f"{y_axis}"].values[0]

            x_values.append(x)
            data_gn.append(y_gn)
            data_louvain.append(y_louvain)
            data_lpa.append(y_lpa)
            data_infomap.append(y_infomap)
            data_gm.append(y_gm)

        data_gn_df = pd.DataFrame(
            {f"{x_axis}": x_values, f"{y_axis}": data_gn}
        )
        data_louvain_df = pd.DataFrame(
            {f"{x_axis}": x_values, f"{y_axis}": data_louvain}
        )
        data_lpa_df = pd.DataFrame(
            {f"{x_axis}": x_values, f"{y_axis}": data_lpa}
        )
        data_infomap_df = pd.DataFrame(
            {f"{x_axis}": x_values, f"{y_axis}": data_infomap}
        )
        data_gm_df = pd.DataFrame(
            {f"{x_axis}": x_values, f"{y_axis}": data_gm}
        )

        return (
            data_gn_df,
            data_louvain_df,
            data_lpa_df,
            data_infomap_df,
            data_gm_df,
        )

    def plot_heatmap(
        self, base_folder, x_axis, y_axis, data_point_values, algorithm, save
    ):
        heatmap_data = self.get_heatmap_data(
            base_folder, x_axis, y_axis, data_point_values, algorithm
        )
        plt.figure(figsize=(20, 5))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidth=0.5)

        # plt.contourf(heatmap_data.columns, heatmap_data.index, heatmap_data.values, 20, cmap="viridis", alpha=0.6)
        # plt.colorbar(label=data_point_values)

        plt.gca().invert_yaxis()

        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(
            f"Heatmap of {data_point_values} with {x_axis} "
            + f"and {y_axis} for {algorithm}"
        )
        plt.tight_layout()

        if save:
            base_dir = os.path.join(f"exports/{base_folder}/plots/")
            filename = (
                base_dir
                + f"heatmap-{data_point_values}-with-{x_axis}-and-{y_axis}"
                + f"-for-{algorithm}-{self.name}.png"
            )
            os.makedirs(base_dir, exist_ok=True)
            plt.savefig(filename, dpi=300)
        plt.show()
        plt.clf()

    def plot_line_chart(self, base_folder, x_axis, y_axis, save):
        (
            data_gn,
            data_louvain,
            data_lpa,
            data_infomap,
            data_gm,
        ) = self.get_line_chart_data(base_folder, x_axis, y_axis)

        data_gn = data_gn.sort_values(by=f"{x_axis}")
        data_louvain = data_louvain.sort_values(by=f"{x_axis}")
        data_lpa = data_lpa.sort_values(by=f"{x_axis}")
        data_infomap = data_infomap.sort_values(by=f"{x_axis}")
        data_gm = data_gm.sort_values(by=f"{x_axis}")

        plt.figure(figsize=(10, 6))

        plt.plot(
            data_gn[f"{x_axis}"],
            data_gn[f"{y_axis}"],
            "ro--",
            label="Girvan Newman",
        )
        plt.plot(
            data_louvain[f"{x_axis}"],
            data_louvain[f"{y_axis}"],
            "go--",
            label="Louvain",
        )
        plt.plot(
            data_lpa[f"{x_axis}"],
            data_lpa[f"{y_axis}"],
            "co--",
            label="LPA",
        )
        plt.plot(
            data_infomap[f"{x_axis}"],
            data_infomap[f"{y_axis}"],
            "go--",
            label="Infomap",
        )
        plt.plot(
            data_gm[f"{x_axis}"],
            data_gm[f"{y_axis}"],
            "mo--",
            label="Greedy Modularity",
        )

        plt.axis.set(xlim=(0, 100), ylim=(0, 1))
        plt.xlabel(f"{x_axis}")
        plt.ylabel(f"{y_axis}")
        plt.title(f"Devolopment of {y_axis} over {x_axis}")
        plt.legend()
        plt.grid(True)

        if save:
            base_dir = os.path.join(f"exports/{base_folder}/plots/")
            filename = (
                base_dir + f"line-chart-{y_axis}-over-{x_axis}-{self.name}.png"
            )
            os.makedirs(base_dir, exist_ok=True)
            plt.savefig(filename, dpi=300)
        plt.show()
        plt.clf()
