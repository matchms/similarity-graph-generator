import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Plots:
    def __init__(self, name):
        self.update_name(name)

    def generate_regex(self, name):
        pattern_str = re.escape(name)
        pattern_str = pattern_str.replace(r"\#", r"(\d+|\d\.\d)")
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
        if plot_type == "confidence":
            self.plot_confidence(base_folder, algorithm, x_axis, y_axis, save)
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
                y_values.append(0) if not y else y_values.append(y)
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

        if all(x is None for x in data_gn):
            data = pd.DataFrame(
                {
                    f"{x_axis}": x_values,
                    "LPA": data_lpa,
                    "Infomap": data_infomap,
                    "GM": data_gm,
                    "Louvain": data_louvain,
                }
            )
        else:
            data = pd.DataFrame(
                {
                    f"{x_axis}": x_values,
                    "LPA": data_lpa,
                    "Infomap": data_infomap,
                    "GM": data_gm,
                    "Louvain": data_louvain,
                    "GN": data_gn,
                }
            )

        consolidated_df = pd.melt(data, [f"{x_axis}"])

        return consolidated_df

    def get_confidence_chart_data(
        self, base_folder, algorithm, x_axis, y_axis
    ):
        data = []

        for run_folder in range(0, 25, 1):
            folder_regex = re.compile(f"^{run_folder}Type.*$")
            for root, dirs, files in os.walk(base_folder):
                for dir in dirs:
                    if folder_regex.match(dir):
                        target_folder = os.path.join(root, dir)
                        for sub_root, sub_dirs, sub_files in os.walk(
                            target_folder
                        ):
                            for file in sub_files:
                                if self.name_pattern.match(file):
                                    csv_path = os.path.join(sub_root, file)
                                    df = pd.read_csv(csv_path)

                                    x = df[f"{x_axis}"][0]
                                    algorithm_row = df[df["Name"] == algorithm]

                                    if not algorithm_row.empty:
                                        y = algorithm_row[f"{y_axis}"].values[
                                            0
                                        ]
                                        data.append(
                                            {
                                                "run": run_folder,
                                                f"{x_axis}": x,
                                                f"{y_axis}": y,
                                            }
                                        )

        consolidated_df = pd.DataFrame(data)
        consolidated_df = consolidated_df.sort_values(
            by=["run", f"{x_axis}"]
        ).reset_index(drop=True)

        return consolidated_df

    def plot_heatmap(
        self, base_folder, x_axis, y_axis, data_point_values, algorithm, save
    ):
        heatmap_data = self.get_heatmap_data(
            base_folder, x_axis, y_axis, data_point_values, algorithm
        )
        plt.figure(figsize=(20, 5))
        sns.heatmap(
            heatmap_data, annot=True, fmt=".2f", linewidth=0.5, vmin=0, vmax=1
        )

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
        else:
            plt.show()
        plt.clf()

    def plot_line_chart(self, base_folder, x_axis, y_axis, save):
        (data) = self.get_line_chart_data(base_folder, x_axis, y_axis)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        sns.lineplot(
            data=data,
            x=x_axis,
            y="value",
            hue="variable",
            style="variable",
            markers=True,
            dashes=True,
        )

        ax.set(xlim=(0, 100), ylim=(0, 1))
        ax.set_yticks(np.arange(0, 1, 1 / 10))
        ax.set_yticks(np.arange(0, 1, 1 / 50), minor=True)
        ax.set_xticks(np.arange(0, 100, 100 / 10))
        plt.xlabel(f"{x_axis}")
        plt.ylabel(f"{y_axis}")
        plt.legend()
        plt.grid(True)

        if save:
            base_dir = os.path.join(f"exports/{base_folder}/plots/")
            filename = (
                base_dir + f"line-chart-{y_axis}-over-{x_axis}-{self.name}.png"
            )
            os.makedirs(base_dir, exist_ok=True)
            plt.savefig(filename, dpi=300)
        else:
            plt.show()
        plt.clf()

    def plot_confidence(self, base_folder, algorithm, x_axis, y_axis, save):
        (consolidated_df) = self.get_confidence_chart_data(
            base_folder, algorithm, x_axis, y_axis
        )

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        sns.lineplot(data=consolidated_df, x=x_axis, y=y_axis)

        ax.set(xlim=(0, 100), ylim=(0, 1))
        ax.set_yticks(np.arange(0, 1, 1 / 10))
        ax.set_yticks(np.arange(0, 1, 1 / 50), minor=True)
        ax.set_xticks(np.arange(0, 100, 100 / 10))
        plt.xlabel(f"{x_axis}")
        plt.ylabel(f"{y_axis}")
        plt.legend()
        plt.grid(True)

        if save:
            base_dir = os.path.join(f"exports/{base_folder}/plots/")
            filename = (
                base_dir
                + f"confidence-{y_axis}-over-{x_axis}-algorithm-{algorithm}-{self.name}.png"
            )
            os.makedirs(base_dir, exist_ok=True)
            plt.savefig(filename, dpi=300)
        else:
            plt.show()
        plt.clf()
