from types import SimpleNamespace

import networkx as nx
from networkx.algorithms.community import modularity
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
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

        - Homogeneity: Provides information if each cluster contains only
          members of single class.
        - Completenes: Provides information if all members if all members of
          class are assigned to same cluster.
        - Community Size: Provides several informations about the number and
          size of the detected communities.
        - Modularity: Measures strength of division of the graph into
          communities.

        - Adjusted Rand Index: Measure similarity between true and predicted
          labels of nodes, adjusted for chance.
        - Normalized Mutual Information: Measure similarity between true and
          predicted labels of nodes.
        - Adjusted Mutual Information: Measure similarity between true and
          predicted labels of nodes, adjusted for chance.
        - Fowlkes Mallows Index: Geometric mean of precision & recall.
        Returns:
            str: Formatted multi-line string giving an overview over the
                 different scores of a community partition of a graph.
        """
        result = (
            f"+++ SCORES FOR {self.name.upper()} +++\n"
            f"+++ Quality metrics +++\n"
            f"Homogeneity: {self.homogeneity:.2f}\n"
            f"Completeness: {self.completeness:.2f}\n"
            f"Community Size: {self.community_size.score:.2f} with "
            f"{self.community_size.num_com:.2f} communities and "
            f"{self.community_size.avg_com_size:.2f} avg community size\n"
            f"Modularity: {self.modularity:.2f}\n"
            f"+++ Clustering metrics +++\n"
            f"Adjusted Rand Index: {self.ari:.2f}\n"
            f"Normalized Mutual Information: {self.nmi:.2f}\n"
            f"Adjusted Mutual Information: {self.ami:.2f}\n"
            f"Fowlkes Mallows Index: {self.fmi:.2f}\n"
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
        self.calc_completeness()
        self.calc_community_size()
        self.calc_modularity()

        self.calc_adjusted_rand_index()
        self.calc_normalized_mutual_information()
        self.calc_adjusted_mutual_information()
        self.calc_fowlkes_mallows()
        return self

    """
    QUALITY METRICS
    """

    def calc_homogeneity(self):
        """
        Calculate the homogeneity score between true and predicted
        labels using 'homogeneity_score' function from scikit-learn.
        Clustering result satisfies homogeneity when each cluster
        conatains only members of single class.
        Score between [0,1]:
        1 -> indicates perfect homogeneous labeling
        """
        self.homogeneity = homogeneity_score(
            self.true_labels, self.predicted_labels
        )

    def calc_completeness(self):
        """
        Calculate the completeness score between true and predicted
        labels using 'completeness_score' function from scikit-learn.
        Clustering result satisfies homogeneity when all members of
        given class assigned to same cluster.
        Score between [0,1]:
        1 -> indicates perfect complete labeling
        """
        self.completeness = completeness_score(
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

    """
    CLUSTERING METRICS
    """

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

    def calc_adjusted_mutual_information(self):
        """
        Calculate adjusted mutual information score between true and
        predicted labels using the 'adjusted_mutual_info_score' function
        from scikit-learn.
        AMI used to measure similarity between two sets of labels, ground
        truth labels (true labels) and the community detection results
        (predicted labels).
        Adjusted for chance -> takes into account that MI generally higher
        for clusterings with larger number of clusters, even if not more
        information is shared.
        1 -> indicates perfect match between label sets
        0 -> indicates random labeling
        negative values -> indicate worse-than-random labeling
        """
        self.ami = adjusted_mutual_info_score(
            self.true_labels, self.predicted_labels
        )

    def calc_fowlkes_mallows(self):
        """
        Calculate the fowlkes mallows index score between the true and
        predicted labels using the 'fowlkes_mallows_score' function from
        scikit-learn.
        FMI defined as geometric mean of precision & recall.
        Score between [0,1]:
        higher value -> good similarity between two clusters
        """
        self.fmi = fowlkes_mallows_score(
            self.true_labels, self.predicted_labels
        )

    def __set_labels(self):
        """
        Set the true and predicted labels for the nodes in the graph based on
        their types and community partitioning. Assigns integer labels to node
        types and communities to allow comparissons between the true labels
        (based on node types) and predicted labels (based on community
        partitioning).
        """
        type_to_int = {
            type_label: type_label
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
