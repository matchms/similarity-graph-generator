import math
from collections import defaultdict
from types import SimpleNamespace

import networkx as nx
import numpy as np
from networkx.algorithms.community import modularity
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)


class Scores:
    def __init__(self, name, graph, subgraphs, partition_list):
        """
        Initialize Score object with a name, graph, subgraphs, and partition
        list.

        Args:
            name (str):             Name of the Score, should have a
                                    correlation to the used algorithm and
                                    applied modifiactions in order to identify
                                    the scores affiliation.
            graph (graph):          The original graph, to which the community
                                    detection algorithm was applied.
            subgraphs (dictionary): Dictionary of subgraphs with community ids
                                    as keys, and the corresponding subgraph as
                                    value.
            partition_list (list):  List of sets representing the detected
                                    communities with nodes of one community in
                                    one set.
        """
        self.name = name
        self.graph = graph
        self.subgraphs = subgraphs
        self.partition = partition_list
        self.__set_labels()
        self.calculate_scores()

    def __str__(self):
        """
        Return a formatted string providing information over the calculated
        scores for a given graphs partition. Can be used to evaluate and
        compare the community detection performance.

        - Homogeneity: Average percentage of most common node type per
          community.
        - Weighted Homogeneity: Like Homogeneity score but takes all node
          types of community into account not just most common one.
        - Normalized Mutual Information: Measure similarity between true and
          predicted labels of nodes.
        - Community Size: Provides several informations about the number and
          size of the detected communities.
        - Adjusted Rand Index: Measure similarity between true and predicted
          labels of nodes, adjusted for chance.
        - F1: The harmonic mean of precision and recall, reperesenting
          accuracy.
        - Modularity: Measures strength of division of the graph into
          communities.
        - Conductance: The average conductance score, measuring the quality of
          the communities.
        - Coverage: Indicates how much of the total connections are within the
          detected communities.

        Returns:
            str: Formatted multi-line string giving an overview over the
                 different scores of a community partition of a graph.
        """
        result = (
            f"+++ SCORES FOR {self.name.upper()} +++\n"
            f"Homogeneity: {self.homogeneity:.2f}\n"
            f"Weighted Homogeneity: {self.weighted_homogeneity:.2f}\n"
            f"Normalized Mutual Information: {self.nmi:.2f}\n"
            f"Community Size: {self.community_size.score:.2f} with "
            f"{self.community_size.num_com:.2f} communities and "
            f"{self.community_size.avg_com_size:.2f} avg community size\n"
            f"Adjusted Rand Index: {self.ari:.2f}\n"
            f"F1: {self.f1:.2f}\n"
            f"Modularity: {self.modularity:.2f}\n"
            f"Conductance: {self.avg_conductance:.2f}\n"
            f"Coverage: {self.coverage:.2f}\n"
        )
        return result

    def calculate_scores(self):
        """
        Calculate all available scores for comparison of a given community
        detection partition.

        Returns:
            self: Instance witch calculated score attributes.
        """
        self.calc_homogeneity()
        self.calc_weighted_homogeneity()
        self.calc_normalized_mutual_information()
        self.calc_community_size()
        self.calc_adjusted_rand_index()
        self.calc_f1_score()
        self.calc_modularity()
        self.calc_conductance()
        self.calc_coverage()
        return self

    def calc_homogeneity(self):
        """
        Calculate the homogeneity for each community in a graph by identifying
        most common node type within each community.
        Homogeneity score for each community is defined as percentage of nodes
        in that community which belong to most common node type.
        The total homogeneity score is defined as average of all computed
        homogeneity scoroes.
        """
        community_scores = []

        for community_id, subgraph in self.subgraphs.items():
            type_counts = defaultdict(int)
            for node in subgraph:
                node_type = self.graph.nodes[node]["type"]
                type_counts[node_type] += 1

            # Calculate score based on most common community type
            most_common_type_count = max(type_counts.values())
            score = (most_common_type_count / len(subgraph)) * 100

            community_scores.append(score)

        all_scores = 0
        for i, score in enumerate(community_scores):
            all_scores += score

        self.homogeneity = all_scores / len(community_scores)

    def calc_weighted_homogeneity(self):
        """
        Calculate the weighted homogeneity for each community in a graph by
        computing the distribution of node types within each community based
        on entropy.
        High entropy -> great node type diversity in community
        Low entropy -> more homogeneous distribution
        Entropy score normalized to [0, 100], with 100 representing perfect
        homogeneity, the total weighted homogeneity score is defined as average
        of all computed homogeneity scores.
        """
        community_scores = []

        for community_id, subgraph in self.subgraphs.items():
            type_counts = defaultdict(int)
            for node in subgraph:
                node_type = self.graph.nodes[node]["type"]
                type_counts[node_type] += 1

            total_nodes = len(subgraph)
            entropy = 0
            for count in type_counts.values():
                proportion = count / total_nodes
                entropy -= proportion * math.log(proportion, 2)

            # Normalize entropy to [0,100]
            max_entropy = math.log(len(type_counts), 2)
            if max_entropy > 0:
                score = (1 - (entropy / max_entropy)) * 100
            else:
                score = 100

            community_scores.append(score)

        total_score = 0
        for i, score in enumerate(community_scores):
            total_score += score

        self.weighted_homogeneity = total_score / len(community_scores)

    def calc_normalized_mutual_information(self):
        """
        Calculate normalized mutual information score between true and
        predicted labels using the 'normalized_mutual_info_score' function
        from scikit-learn.
        NMI used to measure similarity between two sets of labels, ground
        truth labels (true labels) and the community detection results
        (predicted labels).
        Normalized to [0,1]:
        0 -> no mutual information (completely different labels)
        1 -> indicates perfect match between label sets
        """
        self.nmi = normalized_mutual_info_score(
            self.true_labels, self.predicted_labels
        )

    def calc_community_size(self):
        """
        Calculates and provides several informations about the size of the
        detected communities, saved in a SimpleNamespace.

        1. num_com: The total number of communities.
        2. com_size: A list of number of nodes in each community.
        3. avg_com_size: The average size of communities.
        4. score: Score that reflects the relation between number of
                  communities and their average size.
        """
        num_communities = len(self.subgraphs)
        community_sizes = [
            len(community) for i, community in self.subgraphs.items()
        ]
        average_community_size = sum(community_sizes) / num_communities

        size_score = average_community_size / num_communities

        self.community_size = SimpleNamespace(
            num_com=num_communities,
            com_size=community_sizes,
            avg_com_size=average_community_size,
            score=size_score,
        )

    def calc_adjusted_rand_index(self):
        """
        Calculate the adjusted rand index score between the true and predicted
        labels using the 'adjusted_rand_score' function from scikit-learn.
        ARI used to measure similarity between two sets of labels, true and
        predicted, and adjusted for chance -> takes into account that certain
        degree of coincidence may occur.
        Score between [-0.5,1]:
        1 -> indicates perfect match between label sets
        0 -> indicates random labeling
        negative values -> indicate worse-than-random labeling
        """
        self.ari = adjusted_rand_score(self.true_labels, self.predicted_labels)

    def calc_f1_score(self):
        """
        Calculate the weighted F1 score between the true and predicted labels
        using the 'f1_score' function from scikit-learn. Can be interpreted as
        a harmonic mean of the precision and recall, where an F1 score reaches
        its best value at 1 and worst score at 0.

        TODO: check wie sinnvoll? original_score 0.21, warum?
        """
        self.f1 = f1_score(
            self.true_labels, self.predicted_labels, average="weighted"
        )

    def calc_modularity(self):
        """
        Calculate the modularity score for the current partition of the graph
        using networkx 'modularity' function. Evaluates the partition of the
        graph into communities by analyzing inter- and intra-community edges.
        1 -> many intra-community edges, but sparse inter, resulting in good
             community structures
        0 -> few intra-community, but many inter-community edges
        """
        self.modularity = modularity(self.graph, self.partition)

    def calc_conductance(self):
        """
        Calculate the average conductance score over all communities in the
        graph. Measures how well community is separated from the rest of the
        graph, with lower values indicating better-defined communities.
        Conductance defined as ratio of number of edges that connect the
        community to nodes outside of it (cut edges) to the smaller of the
        total volume of the community (sum of the degrees of its nodes)
        and the total volume of the graph minus the volume of the community.
        """
        conductance_scores = []
        for community in self.partition:
            nodes = set(community)
            sum_degree = sum(self.graph.degree(n) for n in nodes)
            cut_edges = nx.cut_size(self.graph, nodes)
            denominator = min(sum_degree, 2 * len(self.graph.edges) - sum_degree)
            conductance = cut_edges / denominator if denominator > 0 else 0
            conductance_scores.append(conductance)
        self.avg_conductance = np.mean(conductance_scores)

    def calc_coverage(self):
        """
        Calculate the coverage of the graph's communities, measuring the
        proportion of edges in the graph that are contained within the
        identified communities. Coverage defined as the ratio of the total
        number of edges that connect nodes within the same community
        (intra-community edges) to the total number of edges in the graph.
        """
        intra_community_edges = sum(
            nx.subgraph(self.graph, community).size()
            for community in self.partition
        )
        total_edges = len(self.graph.edges)
        self.coverage = intra_community_edges / total_edges

    def __set_labels(self):
        """
        Set the true and predicted labels for the nodes in the graph based on
        their types and community partitioning. Assigns integer labels to node
        types and communities to allow comparissons between the true labels
        (based on node types) and predicted labels (based on community
        partitioning).
        """
        type_to_int = {
            type_label: i
            for i, type_label in enumerate(
                set(nx.get_node_attributes(self.graph, "type").values())
            )
        }
        self.true_labels = [
            type_to_int[self.graph.nodes[node]["type"]]
            for node in self.graph.nodes()
        ]

        self.predicted_labels = [None] * len(self.graph.nodes())
        for label, community in enumerate(self.partition):
            for node in community:
                self.predicted_labels[node] = label
