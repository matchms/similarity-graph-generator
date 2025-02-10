import copy
import os
from collections import defaultdict

import networkx as nx
import numpy as np
from community import community_louvain
from infomap import Infomap
from matplotlib import pyplot as plt
from networkx.algorithms.community import girvan_newman


class Graph:
    def __init__(self, similarity_matrix):
        """
        Constructor, initializes instance of Graph class.
        Initializes an "original_untouched_graph" representing
        the graph based on the provided similarity matrix before
        applying any modifications.

        Args:
            similarity_matrix: a matrix of similiarity measures
        """
        self.options = {
            "matrix": {
                "threshold": False,
                "normalization": False,
            },
            "graph": {
                "edge_threshold_node_based": False,
                "min_edges_per_node_node_based": False,
                "edge_threshold_global": False,
                "min_edges_per_node_global": False,
                "edge_weight_normalization": False,
            },
            "community_detection": {
                "girvan_newman": False,
                "louvain": False,
                "lpa": False,
                "infomap": False,
                "greedy_modularity": False,
            },
        }
        self.similarity_matrix = similarity_matrix
        self.original_untouched_graph = nx.from_numpy_array(similarity_matrix)

    """
    MATRIX
    """

    def apply_matrix_treshold(self, percentage_to_remove=20):
        """
        Modify the similarity matrix on that the graph will be based.
        This method can be used to remove low similarities in order to have
        less edges later in the graph. This helps to focus on stronger
        connections between nodes.
        For example: if the provided percentage_to_remove would be 20, then the
        lowest 20% similarities will be set to 0.

        Args:
            percentage_to_remove (int):   Provides the percentile below which
                                            the values in the similarity matrix
                                            are set to 0.
                                            Should be in range [0, 100]
        """
        similarities_flat = self.similarity_matrix.flatten()
        threshold = np.percentile(similarities_flat, percentage_to_remove)
        self.similarity_matrix[self.similarity_matrix < threshold] = 0
        self.options["matrix"]["threshold"] = percentage_to_remove

    def apply_matrix_normalization(self):
        """
        This function normalizes the values of the similarity matrix
        to a range of [0,1]. This ensures that the algorithms that are later
        applied to the graph have the best possible performace in case the
        values of the provided similarity matrix were not normalized before.
        Also if a matrix treshold was applied beforehand, with
        apply_matrix_threshold function, it is likely that the values are in
        a range between [0.5,1] for example. Normalizing the values after that
        helps to improve the further processing and visualization in the end.
        """
        # remove zeros to get next smallest value
        array = np.array(self.similarity_matrix, dtype=float)
        non_zero_values = array[array != 0]
        min = np.min(non_zero_values)
        max = np.max(non_zero_values)

        # Normalize non-zero values
        normalized_matrix = array.copy()
        normalized_matrix[array != 0] = (array[array != 0] - min) / (max - min)

        self.options["matrix"]["normalization"] = True
        self.similarity_matrix = normalized_matrix

    """
    GRAPH
    """

    def create_graph(self):
        """
        Generate a graph based on the provided similarity matrix.
        Currently with compounds (to be removed later):
        Nodes represent compounds, ean edge represents the similarity score
        between two compounds. The edge weight expresses the similarity score
        from the matrix. The nodes get assigned attributes based on the
        properties of the compounds.
        """
        self.graph = nx.from_numpy_array(self.similarity_matrix)

    def apply_edge_treshold_node_based(
        self, percentage_to_remove=20, min_edges_per_node=0
    ):
        """
        Reduce the number of edges per node based on the weight of the edges
        and the given threshold percentage_to_remove. To avoid nodes scattered
        all over the place without edges connecting them; min_edges_per_node
        ensures that each node keeps at least a given number of edges, even
        though the edge might be of a lower weight.

        The function sorts the edges of each node by weight, then calculates
        based on percentage_to_remove how many edges should be removed for this
        node. Then the edges are removed from the graph as long, as the
        calculated number of edges to remove is not met and both nodes,
        connected by the current edge, have more than min_edges_per_node edges
        left.

        Args:
            percentage_to_remove (int):   Provides the percentage of edges
                                            that should be removed from a node.
                                            Should be in range [0, 100]
            min_edges_per_node (int):       Defines how many edges a node
                                            should be left with. Default is 0,
                                            should be be a positive Integer.
        """
        graph = copy.deepcopy(self.graph)

        for node in graph.nodes:
            edges = list(graph.edges(node, data=True))
            edges_sorted = sorted(edges, key=lambda x: x[2]["weight"])
            num_edges_to_remove = int(
                len(edges_sorted) * (percentage_to_remove / 100)
            )

            for edge in edges_sorted:
                if num_edges_to_remove > 0:
                    u, v, _ = edge
                    if (
                        graph.degree(u) - 1 >= min_edges_per_node
                        and graph.degree(v) - 1 >= min_edges_per_node
                    ):
                        graph.remove_edge(u, v)
                        num_edges_to_remove -= 1

        self.options["graph"][
            "edge_threshold_node_based"
        ] = percentage_to_remove
        self.options["graph"][
            "min_edges_per_node_node_based"
        ] = min_edges_per_node
        self.graph = graph

    def apply_edge_threshold_global(
        self, percentage_to_remove=20, min_edges_per_node=0
    ):
        """
        Reduce the global number of edges in the graph based on the weight of
        the edges and the given threshold percentage_to_remove. To avoid nodes
        scattered all over the place without edges connecting them;
        min_edges_per_node ensures that each node keeps at least a given number
        of edges, even though the edge might be of a lower weight.

        Args:
            percentage_to_remove (int):   Provides the percentage of edges
                                            that should be removed from the
                                            graph. Should be in range
                                            [0, 100]
            min_edges_per_node (int):       Defines how many edges a node
                                            should be left with. Default
                                            is 0, should be be a positive
                                            Integer.
        """
        graph = copy.deepcopy(self.graph)
        num_edges_to_remove = int(
            graph.number_of_edges() * (percentage_to_remove / 100)
        )
        edges_sorted = sorted(
            graph.edges(data=True), key=lambda x: x[2]["weight"]
        )
        edges_removed = 0

        for u, v, data in edges_sorted:
            if (graph.degree(u) - 1 >= min_edges_per_node) and (
                graph.degree(v) - 1 >= min_edges_per_node
            ):
                edges_removed += 1
                graph.remove_edge(u, v)

                if edges_removed >= num_edges_to_remove:
                    break

        self.options["graph"]["edge_threshold_global"] = percentage_to_remove
        self.options["graph"]["min_edges_per_node_global"] = min_edges_per_node
        self.graph = graph

    def apply_edge_weight_normalization(self):
        """
        Normalize edge weights to vales between 0 and 1.
        """
        edge_weights = [
            data["weight"] for _, _, data in self.graph.edges(data=True)
        ]

        min_weight = min(edge_weights)
        max_weight = max(edge_weights)

        for u, v, data in self.graph.edges(data=True):
            data["weight"] = (data["weight"] - min_weight) / (
                max_weight - min_weight
            )

        self.options["graph"]["edge_weight_normalization"] = True

    """
    COMMUNITY DETECTION ALGORITHMS
    """

    def apply_girvan_newman(self):
        """
        Apply Girvan-Newman algorithm to the graph. Generates a partition_list
        with nodes grouped into sets based on the detected communities, a graph
        where the edges between detected communities are removed and a
        dictionary of subgraphs for each community.
        """
        com_itr = girvan_newman(self.graph)

        self.girvan_newman_partition_list = list(
            sorted(c) for c in next(com_itr)
        )
        self.girvan_newman_graph = self.__create_cd_graph(
            self.girvan_newman_partition_list
        )
        self.girvan_newman_partitioned_graph = self.__create_partitioned_graph(
            self.girvan_newman_partition_list
        )
        self.girvan_newman_weight_adjusted_graph = (
            self.__adjust_edge_weights_community_based(
                self.girvan_newman_partition_list
            )
        )
        self.girvan_newman_subgraphs = self.__create_subgraphs(
            self.girvan_newman_partition_list
        )
        self.options["community_detection"]["girvan_newman"] = True

    def apply_louvain(self):
        """
        Apply Louvain algorithm to the graph. Generates a partition_list
        with nodes grouped into sets based on the detected communities, a graph
        where the edges between detected communities are removed and a
        dictionary of subgraphs for each community.
        """
        self.louvain_partition = community_louvain.best_partition(self.graph)
        type_to_nodes = defaultdict(set)
        for node, node_type in self.louvain_partition.items():
            type_to_nodes[node_type].add(node)

        self.louvain_partition_list = list(type_to_nodes.values())
        self.louvain_graph = self.__create_cd_graph(
            self.louvain_partition_list
        )
        self.louvain_partitioned_graph = self.__create_partitioned_graph(
            self.louvain_partition_list
        )
        self.louvain_weight_adjusted_graph = (
            self.__adjust_edge_weights_community_based(
                self.louvain_partition_list
            )
        )
        self.louvain_subgraphs = self.__create_subgraphs(
            self.louvain_partition_list
        )
        self.options["community_detection"]["louvain"] = True

    def apply_lpa(self):
        """
        Apply Label-Propagation algorithm to the graph. Generates a
        partition_list with nodes grouped into sets based on where the edges
        between detected communities are removed and a dictionary of subgraphs
        for each community.
        """
        lpa_partition = list(
            nx.algorithms.community.label_propagation_communities(self.graph)
        )

        self.lpa_partition_list = lpa_partition
        self.lpa_graph = self.__create_cd_graph(self.lpa_partition_list)
        self.lpa_partitioned_graph = self.__create_partitioned_graph(
            self.lpa_partition_list
        )
        self.lpa_weight_adjusted_graph = (
            self.__adjust_edge_weights_community_based(self.lpa_partition_list)
        )
        self.lpa_subgraphs = self.__create_subgraphs(self.lpa_partition_list)
        self.options["community_detection"]["lpa"] = True

    def apply_infomap(self):
        """
        Apply Infomap algorithm to the graph. Generates a partition_list
        with nodes grouped into sets based on the detected communities, a graph
        where the edges between detected communities are removed and a
        dictionary of subgraphs for each community.
        """
        infomap = Infomap(silent=True)
        for node in self.graph.nodes():
            infomap.add_node(node)
        infomap.add_links(self.graph.edges())
        infomap.run()
        communities = {}
        for node_id, module_id in infomap.modules:
            if module_id not in communities:
                communities[module_id] = set()
            communities[module_id].add(node_id)

        self.infomap_partition_list = list(communities.values())
        self.infomap_graph = self.__create_cd_graph(
            self.infomap_partition_list
        )
        self.infomap_partitioned_graph = self.__create_partitioned_graph(
            self.infomap_partition_list
        )
        self.infomap_weight_adjusted_graph = (
            self.__adjust_edge_weights_community_based(
                self.infomap_partition_list
            )
        )
        self.infomap_subgraphs = self.__create_subgraphs(
            self.infomap_partition_list
        )
        self.options["community_detection"]["infomap"] = True

    def apply_greedy_modularity(self):
        """
        Apply Greedy-Modularity algorithm to the graph. Generates a
        partition_list with nodes grouped into sets based on the detected
        communities, a graph where the edges between detected communities
        are removed and a dictionary of subgraphs for each community.
        """
        communities = nx.algorithms.community.greedy_modularity_communities(
            self.graph
        )
        self.greedy_modularity_partition_list = [
            set(community) for community in communities
        ]
        self.greedy_modularity_graph = self.__create_cd_graph(
            self.greedy_modularity_partition_list
        )
        self.greedy_modularity_partitioned_graph = (
            self.__create_partitioned_graph(
                self.greedy_modularity_partition_list
            )
        )
        self.greedy_modularity_weight_adjusted_graph = (
            self.__adjust_edge_weights_community_based(
                self.greedy_modularity_partition_list
            )
        )
        self.greedy_modularity_subgraphs = self.__create_subgraphs(
            self.greedy_modularity_partition_list
        )
        self.options["community_detection"]["greedy_modularity"] = True

    def __create_cd_graph(self, partition):
        """
        Creates a copy of the graph to that the community detection
        alorithm was applied and changes the color of the edges.
        Inter community edges are set to grey with lower opacity.
        Intra community edges are set to black with full opacity.

        Args:
            partition (list): List of sets representing the detected
                            communities with nodes of one community in one
                            set.

        Returns:
            graph: the graph on that community detection algorithm was
            applied, but with changed edge colors.
        """
        graph = copy.deepcopy(self.graph)
        node_to_community = {}
        for community_id, com in enumerate(partition):
            for node in com:
                node_to_community[node] = community_id

        for u, v, data in graph.edges(data=True):
            community_u = node_to_community.get(u, None)
            community_v = node_to_community.get(v, None)

            if community_u == community_v:
                data["color"] = "#000000"
            else:
                data["color"] = "#4646464d"

        return graph

    def __create_partitioned_graph(self, partition):
        """
        Create a graph based on the partition a community detection algorithm
        generated. Uses the original graph to that the community detection
        algorithm was applied and removes all edges connecting the detected
        communities.

        Args:
            partition (list):   List of sets representing the detected
                                communities with nodes of one community in one
                                set.

        Returns:
            graph:  the graph on that community detection algorithm was
                    applied, but without edges between different communities
        """
        partitioned_graph = copy.deepcopy(self.graph)
        node_to_community = {}

        for community_id, com in enumerate(partition):
            for node in com:
                node_to_community[node] = community_id

        edges_to_remove = [
            (u, v)
            for u, v in partitioned_graph.edges()
            if node_to_community[u] != node_to_community[v]
        ]

        partitioned_graph.remove_edges_from(edges_to_remove)
        return partitioned_graph

    def __adjust_edge_weights_community_based(self, partition):
        """
        Creates a copy of the graph to that the community detection alorithm
        was applied and changes the color and weights of the edges.
        Intra community edges are set to black with full opacity and the edge
        weight is multiplied by 2. Inter community edges are set to grey with
        lower opacity.

        Args:
            partition (list): List of sets representing the detected
                            communities with nodes of one community in one
                            set.

        Returns:
            graph: the graph on that community detection algorithm was applied,
            but with changed edge colors and weights.
        """
        adjusted_graph = copy.deepcopy(self.original_untouched_graph)
        node_to_community = {}
        for community_id, com in enumerate(partition):
            for node in com:
                node_to_community[node] = community_id

        for u, v, data in adjusted_graph.edges(data=True):
            community_u = node_to_community.get(u, None)
            community_v = node_to_community.get(v, None)

            if community_u == community_v:
                data["weight"] *= 2
                data["color"] = "#000000"
            else:
                data["color"] = "#4646464d"

        return adjusted_graph

    def __create_subgraphs(self, partition):
        """
        Create multiple subgraphs from the original graph, based on the
        detected communities provided by the partition.

        Args:
            partition (list):   List of sets representing the detected
                                communities, with nodes of one community in
                                one set.

        Returns:
            dictionary: keys are community ids, values are the corresponding
                        subgraph
        """
        subgraphs = {}

        for community_id, community_nodes in enumerate(partition):
            subgraph = nx.Graph()
            subgraph.add_nodes_from(community_nodes)

            for u, v, data in self.original_untouched_graph.edges(data=True):
                if u in community_nodes and v in community_nodes:
                    subgraph.add_edge(u, v, weight=data["weight"])

            for node in subgraph.nodes():
                nx.set_node_attributes(
                    subgraph, {node: self.original_untouched_graph.nodes[node]}
                )

            subgraphs[community_id] = subgraph

        return subgraphs

    """
    VISUALIZATION
    """

    def visualize_similarities_histogram(
        self,
        similarity_matrix,
        folder_name=None,
        show=True,
        save=False,
        name="histogram.png",
    ):
        """
        Visualize a histogram of the values in the similarity matrix. Zero
        values are not displayed.
        Args:
            show (bool):   Histogram will be shown if True.
            save (bool):   Histogram will be saved as png if True.
        """
        similarities_flat = similarity_matrix[similarity_matrix != 0].flatten()
        plt.hist(similarities_flat, bins=50, rwidth=0.8)

        if save:
            name_as_code = self.__get_name_as_code()
            if folder_name:
                base_dir = os.path.join(
                    f"exports/{folder_name}/{name_as_code}/images/"
                )
            else:
                base_dir = os.path.join(f"exports/{name_as_code}/images/")
            os.makedirs(base_dir, exist_ok=True)
            plt.savefig(base_dir + name, dpi=100)
        if show:
            plt.show()
        plt.clf()

    def visualize_graph(
        self,
        graph_to_plot=None,
        title=None,
        node_distance=None,
        show_edge_weights=False,
        show=True,
        save=False,
    ):
        """
        Visualizes a graph. If no graph is provided, the original graph is
        displayed. The function uses spring_layout for a more appealing
        visualization.

        Args:
            graph_to_plot (graph, optional):    The graph to plot. If none
                                                provided, original graph is
                                                displayed.
            title (str, optional):              Title of the plot. If not
                                                provided a name is generated
                                                from the option dictionary.
            node_distance (float, optional):    Optimal distance between
                                                nodes. Increase this value to
                                                move nodes farther apart.
            show_edge_weights (bool, optional): If true, edges are labeled with
                                                their weight value. Defaults to
                                                False.
            show (bool):                        Graph will be shown if True.
            safe (bool):                        Graph will be saved as png if
                                                True.
        """
        graph = graph_to_plot if graph_to_plot else self.graph
        name = title if title else self.__get_name()
        distance = node_distance if node_distance else 0.05

        pos = nx.spring_layout(graph, k=distance, seed=0)
        labels = {node: str(node) for node in graph.nodes()}
        edge_labels = {
            (u, v): f"{data['weight']:.2f}"
            for u, v, data in graph.edges(data=True)
        }
        edge_colors = (
            nx.get_edge_attributes(graph, "color").values()
            if len(nx.get_edge_attributes(graph, "color").values()) > 0
            else [0, 0, 0]
        )

        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        nx.draw_networkx_edges(
            graph, pos, alpha=0.4, ax=ax, edge_color=edge_colors
        )
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=100,
            ax=ax,
        )
        nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=6)
        if show_edge_weights:
            nx.draw_networkx_edge_labels(
                graph, pos, edge_labels=edge_labels, ax=ax, font_size=6
            )

        ax.set_title(name)
        ax.axis("off")
        if save:
            name_as_code = self.__get_name_as_code()
            base_dir = os.path.join(f"exports/{name_as_code}/images")
            filename = (
                base_dir + "/" + title + ".png"
                if title
                else base_dir + "/" + "graph.png"
            )
            os.makedirs(base_dir, exist_ok=True)
            plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        plt.clf()

    def visualize_partition(
        self,
        subgraphs_to_plot,
        min_nodes=None,
        title=None,
        node_distance=None,
        show_edge_weights=False,
        show=True,
        save=False,
    ):
        """
        Visualizes the subgraphs of a graph representing the communities that
        were found by a community detection algorithm. The subgraphs are
        displayed in a grid layout and can be filtered to only display
        subgraphs that have more nodes than specified by min_nodes.
        The function uses spring_layout for a more appealing visualization.

        Args:
            subgraphs_to_plot (dict):   Dictionary of subgraphs with community
                                        ids as keys, and the corresponding
                                        subgraph as value.
            min_nodes (int):            Minimum number of nodes a subgraph
                                        needs to have, to be displayed.
                                        Defaults to 0, meaning no subgraphs
                                        are excluded.
            title (str):                Title of the plot. If not provided a
                                        name is generated from the option
                                        dictionary.
            node_distance (float):      Optimal distance between nodes.
                                        Increase this value to move nodes
                                        further apart.
            show_edge_weights (bool):   If true, edges are labeled with their
                                        weight value. Defaults to False.
            show (bool):                Subgraphs will be shown if True.
            safe (bool):                Subgraphs will be saved as png if True.
        """
        name = title if title else self.__get_name()
        distance = node_distance if node_distance else 0.05
        node_threshold = min_nodes if min_nodes else 0
        filtered_subgraphs = [
            subgraph
            for subgraph in subgraphs_to_plot.values()
            if len(subgraph.nodes) > node_threshold
        ]

        # Calculate grid layout
        num_subgraphs = len(filtered_subgraphs)
        cols = int(num_subgraphs**0.5) + 1
        rows = (num_subgraphs + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
        axs = axs.flatten()

        for index, graph in enumerate(filtered_subgraphs):
            ax = axs[index]
            pos = nx.spring_layout(graph, k=distance, seed=0)
            labels = {node: str(node) for node in graph.nodes()}
            edge_labels = {
                (u, v): f"{data['weight']:.2f}"
                for u, v, data in graph.edges(data=True)
            }
            nx.draw_networkx_edges(graph, pos, alpha=0.4, ax=ax)
            nx.draw_networkx_nodes(
                graph,
                pos,
                node_size=100,
                ax=ax,
            )
            nx.draw_networkx_labels(
                graph, pos, labels=labels, ax=ax, font_size=6
            )
            if show_edge_weights:
                nx.draw_networkx_edge_labels(
                    graph, pos, edge_labels=edge_labels, ax=ax, font_size=6
                )
            ax.set_title(f"{name} Community: {index}")
            ax.axis("off")

        # Remove unused subplots
        for index in range(num_subgraphs, len(axs)):
            fig.delaxes(axs[index])

        plt.tight_layout()
        if save:
            name_as_code = self.__get_name_as_code()
            base_dir = os.path.join(f"exports/{name_as_code}/images")
            filename = (
                base_dir + "/" + title + ".png"
                if title
                else base_dir + "/" + "subgraphs.png"
            )
            os.makedirs(base_dir, exist_ok=True)
            plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        plt.clf()

    def __get_name(self):
        """
        Generate a name for the graph, based on the options dictionary.
        The name consists of the options, that have be applied to the graph to
        allow a quick understanding how the graph got modified.

        Returns:
            String: the name of the graph
        """
        name = ""
        for category, options in self.options.items():
            true_options = []
            for option, value in options.items():
                if value:
                    true_options.append(option)
            if true_options:
                name += category.swapcase() + ": "
                for option in true_options:
                    name += option + ", "

        return name

    def __get_name_as_code(self):
        """
        Generate a string representing the configuration of the graph.

        Generated string format:
        'node-count'
        _'threshold'-'normalization'
        _'threshold-node-based'-'min_edges_node_based'-'threshold-global'-'min_edges_global'-'weight-normalization'
        _'girvan_newman'-'louvain'-'lpa'-'infomap'-'greedy-modularity'

        example output:
        340_20-1_10-90-5-60-5-0_1-1-1-1-0
        """
        name = ""
        name += str(self.original_untouched_graph.number_of_nodes())
        for category, options in self.options.items():
            name += "_"
            for option, value in options.items():
                if not isinstance(value, bool):
                    name += str(value) + "-"
                else:
                    name += "1-" if value else "0-"
            name = name[:-1]

        return name

    """
    EXPORT
    """

    def export_graphml(self):
        """
        Export the original graph and the graphs and subgraphs of the applied
        community detection algorithms in graphml file for further processing.
        The files are safed in a "graphml" folder and then based on the graphs
        name into further subfolders.
        """
        name_as_code = self.__get_name_as_code()

        base_dir = os.path.join(f"exports/{name_as_code}/graphml")
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "original"), exist_ok=True)

        for option in self.options["community_detection"]:
            if self.options["community_detection"].get(option):
                os.makedirs(os.path.join(base_dir, option), exist_ok=True)

                graph = getattr(self, f"{option}_graph", None)
                partitioned_graph = getattr(
                    self, f"{option}_partitioned_graph", None
                )
                weight_adjusted_graph = getattr(
                    self, f"{option}_weight_adjusted_graph", None
                )
                subgraphs = getattr(self, f"{option}_subgraphs", None)

                nx.write_graphml(
                    graph,
                    f"exports/{name_as_code}/graphml/{option}/"
                    + f"graph_{name_as_code}.graphml",
                )
                nx.write_graphml(
                    partitioned_graph,
                    f"exports/{name_as_code}/graphml/{option}/"
                    + f"partitioned_graph_{name_as_code}.graphml",
                )
                nx.write_graphml(
                    weight_adjusted_graph,
                    f"exports/{name_as_code}/graphml/{option}/"
                    + f"weight_adjusted_graph_{name_as_code}.graphml",
                )
                for community_id, subgraph in subgraphs.items():
                    nx.write_graphml(
                        subgraph,
                        f"exports/{name_as_code}/graphml/{option}/"
                        + f"subgraph_{name_as_code}_community_{community_id}"
                        + ".graphml",
                    )

    def export_all_images(
        self,
        folder_name=None,
        export_histogram=True,
        export_graph=True,
        export_partition=True,
        export_cd=True,
    ):
        """
        Save all images to the export folder. Includes histogram for similarity
        matrix, all generated graphs and subraphs.
        """
        if export_histogram:
            self.visualize_similarities_histogram(
                folder_name=folder_name,
                similarity_matrix=self.similarity_matrix,
                show=False,
                save=True,
            )
        if export_graph:
            self.visualize_graph(self.graph, "graph", show=False, save=True)
        if export_cd:
            for option in self.options["community_detection"]:
                if self.options["community_detection"].get(option):
                    graph = getattr(self, f"{option}_graph", None)
                    partitioned_graph = getattr(
                        self, f"{option}_partitioned_graph", None
                    )
                    weight_adjusted_graph = getattr(
                        self, f"{option}_weight_adjusted_graph", None
                    )
                    subgraphs = getattr(self, f"{option}_subgraphs", None)

                    self.visualize_graph(
                        graph, title=f"{option}_graph", show=False, save=True
                    )
                    self.visualize_graph(
                        partitioned_graph,
                        title=f"{option}_partitioned",
                        show=False,
                        save=True,
                    )
                    self.visualize_graph(
                        weight_adjusted_graph,
                        title=f"{option}_weight_adjusted_graph",
                        show=False,
                        save=True,
                    )
                    self.visualize_partition(
                        subgraphs,
                        title=f"{option}_subgraphs",
                        show=False,
                        save=True,
                    )
