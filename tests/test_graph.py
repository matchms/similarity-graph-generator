import os
from unittest import mock

import numpy as np
import pytest
from similaritygraphgenerator.classes.graph import Graph


@pytest.fixture
def some_matrix():
    matrix = np.array(
        [
            [
                0.0,
                0.67142392,
                0.67489393,
                0.42128391,
                0.50297495,
                0.48906283,
                0.79470758,
                0.43822864,
            ],
            [
                0.67142392,
                0.0,
                0.39038537,
                0.14495787,
                0.81712863,
                0.37301025,
                0.4951345,
                0.28934718,
            ],
            [
                0.67489393,
                0.39038537,
                0.0,
                0.38316204,
                0.51332427,
                0.61081052,
                0.38152867,
                0.18041858,
            ],
            [
                0.42128391,
                0.14495787,
                0.38316204,
                0.0,
                0.43912465,
                0.43010169,
                0.83708672,
                0.83701597,
            ],
            [
                0.50297495,
                0.81712863,
                0.51332427,
                0.43912465,
                0.0,
                0.05450013,
                0.34957277,
                0.54488783,
            ],
            [
                0.48906283,
                0.37301025,
                0.61081052,
                0.43010169,
                0.05450013,
                0.0,
                0.34578905,
                0.56640406,
            ],
            [
                0.79470758,
                0.4951345,
                0.38152867,
                0.83708672,
                0.34957277,
                0.34578905,
                0.0,
                0.57578309,
            ],
            [
                0.43822864,
                0.28934718,
                0.18041858,
                0.83701597,
                0.54488783,
                0.56640406,
                0.57578309,
                0.0,
            ],
        ]
    )
    return matrix


@pytest.fixture
def some_other_matrix():
    matrix = np.array(
        [
            [
                0.0,
                0.67142392,
                0.67489393,
                0.42128391,
                0.50297495,
                0.48906283,
                0.79470758,
                0.43822864,
            ],
            [
                0.67142392,
                0.0,
                0.39038537,
                0.14495787,
                0.81712863,
                0.37301025,
                0.4951345,
                0.28934718,
            ],
            [
                0.67489393,
                0.39038537,
                0.0,
                0.38316204,
                0.51332427,
                0.61081052,
                0.38152867,
                0.18041858,
            ],
            [
                0.42128391,
                0.14495787,
                0.38316204,
                0.0,
                0.43912465,
                0.43010169,
                0.83708672,
                0.83701597,
            ],
            [
                0.50297495,
                0.81712863,
                0.51332427,
                0.43912465,
                0.0,
                0.05450013,
                0.34957277,
                0.54488783,
            ],
            [
                0.48906283,
                0.37301025,
                0.61081052,
                0.43010169,
                0.05450013,
                0.0,
                0.34578905,
                0.56640406,
            ],
            [
                0.79470758,
                0.4951345,
                0.38152867,
                0.83708672,
                0.34957277,
                0.34578905,
                0.0,
                0.57578309,
            ],
            [
                0.43822864,
                0.28934718,
                0.18041858,
                0.83701597,
                0.54488783,
                0.56640406,
                0.57578309,
                0.0,
            ],
        ]
    )
    return matrix


@pytest.fixture
def original_graph(some_matrix):
    graph = Graph(some_matrix)
    return graph


@pytest.fixture
def modified_graph(some_other_matrix):
    graph = Graph(some_other_matrix)
    return graph


def test_apply_no_matrix_treshold(original_graph, modified_graph):
    modified_graph.apply_matrix_treshold(percentage_to_remove=0)
    assert (
        original_graph.similarity_matrix == modified_graph.similarity_matrix
    ).all()


def test_apply_matrix_treshold(original_graph, modified_graph):
    modified_graph.apply_matrix_treshold(percentage_to_remove=80)
    assert not (
        original_graph.similarity_matrix == modified_graph.similarity_matrix
    ).all()


def test_apply_matrix_normalization(original_graph, modified_graph):
    original_graph.apply_matrix_treshold()
    modified_graph.apply_matrix_treshold()
    modified_graph.apply_matrix_normalization()
    assert not (
        original_graph.similarity_matrix == modified_graph.similarity_matrix
    ).all()


def test_create_graph(modified_graph):
    modified_graph.create_graph()
    assert len(modified_graph.graph.nodes()) == 8
    assert len(modified_graph.graph.edges()) == 28


def test_apply_edge_treshold_node_based(original_graph, modified_graph):
    original_graph.create_graph()
    modified_graph.create_graph()
    modified_graph.apply_edge_treshold_node_based(percentage_to_remove=50)
    assert len(modified_graph.graph.edges()) < len(
        original_graph.graph.edges()
    )
    for node in set(modified_graph.graph.nodes()).union(
        original_graph.graph.nodes()
    ):
        assert len(list(modified_graph.graph.edges(node))) < len(
            list(original_graph.graph.edges(node))
        )


def test_apply_edge_threshold_global(modified_graph):
    modified_graph.create_graph()
    modified_graph.apply_edge_threshold_global(percentage_to_remove=50)
    assert len(modified_graph.graph.edges()) == 14


def test_apply_edge_weight_normalization(original_graph, modified_graph):
    original_graph.create_graph()
    modified_graph.create_graph()
    original_graph.apply_edge_threshold_global()
    modified_graph.apply_edge_threshold_global()
    assert len(modified_graph.graph.edges()) == len(
        original_graph.graph.edges()
    )
    for node in set(modified_graph.graph.nodes()).union(
        original_graph.graph.nodes()
    ):
        for neighbor in modified_graph.graph.neighbors(node):
            if modified_graph.graph.has_edge(
                node, neighbor
            ) and original_graph.graph.has_edge(node, neighbor):
                assert (
                    modified_graph.graph[node][neighbor]["weight"]
                    == original_graph.graph[node][neighbor]["weight"]
                )
    modified_graph.apply_edge_weight_normalization()
    for node in set(modified_graph.graph.nodes()).union(
        original_graph.graph.nodes()
    ):
        for neighbor in modified_graph.graph.neighbors(node):
            if modified_graph.graph.has_edge(
                node, neighbor
            ) and original_graph.graph.has_edge(node, neighbor):
                assert (
                    not modified_graph.graph[node][neighbor]["weight"]
                    == original_graph.graph[node][neighbor]["weight"]
                )


def test_apply_girvan_newman(modified_graph):
    modified_graph.create_graph()
    modified_graph.apply_girvan_newman()
    assert modified_graph.options["community_detection"]["girvan_newman"]
    assert len(modified_graph.girvan_newman_partition_list) > 0


def test_apply_louvain(modified_graph):
    modified_graph.create_graph()
    modified_graph.apply_louvain()
    assert modified_graph.options["community_detection"]["louvain"]
    assert len(modified_graph.louvain_partition_list) > 0


def apply_lpa(modified_graph):
    modified_graph.create_graph()
    modified_graph.apply_lpa()
    assert modified_graph.options["community_detection"]["lpa"]
    assert len(modified_graph.lpa_partition_list) > 0


def apply_infomap(modified_graph):
    modified_graph.create_graph()
    modified_graph.apply_infomap()
    assert modified_graph.options["community_detection"]["infomap"]
    assert len(modified_graph.infomap_partition_list) > 0


def apply_greedy_modularity(modified_graph):
    modified_graph.create_graph()
    modified_graph.apply_greedy_modularity()
    assert modified_graph.options["community_detection"]["greedy_modularity"]
    assert len(modified_graph.greedy_modularity_partition_list) > 0


def test_visualize_similarities_histogram_save(modified_graph):
    with (
        mock.patch("matplotlib.pyplot.savefig") as mock_save,
        mock.patch("os.makedirs") as mock_makedirs,
        mock.patch("matplotlib.pyplot.show") as mock_show,
    ):
        modified_graph.visualize_similarities_histogram(
            modified_graph.similarity_matrix, show=False, save=True
        )
        base_dir = os.path.join("exports/8_0-0_0-0-0-0-0_0-0-0-0-0/images/")
        mock_makedirs.assert_called_with(base_dir, exist_ok=True)
        name = "histogram.png"
        mock_save.assert_called_with(base_dir + name, dpi=100)
        mock_show.assert_not_called()


def test_visualize_similarities_histogram_show(modified_graph):
    with (
        mock.patch("matplotlib.pyplot.savefig") as mock_save,
        mock.patch("os.makedirs") as mock_makedirs,
        mock.patch("matplotlib.pyplot.show") as mock_show,
    ):
        modified_graph.visualize_similarities_histogram(
            modified_graph.similarity_matrix, show=True, save=False
        )
        mock_show.assert_called_once()
        mock_save.assert_not_called()
        mock_makedirs.assert_not_called()


def test_visualize_graph_save(modified_graph):
    with (
        mock.patch("matplotlib.pyplot.savefig") as mock_save,
        mock.patch("os.makedirs") as mock_makedirs,
        mock.patch("matplotlib.pyplot.show") as mock_show,
    ):
        modified_graph.create_graph()
        modified_graph.visualize_graph(
            modified_graph.graph, show=False, save=True, title="test"
        )
        base_dir = os.path.join("exports/8_0-0_0-0-0-0-0_0-0-0-0-0/images")
        mock_makedirs.assert_called_with(base_dir, exist_ok=True)
        name = "/test.png"
        mock_save.assert_called_with(base_dir + name, dpi=300)
        mock_show.assert_not_called()


def test_visualize_graph_show(modified_graph):
    with (
        mock.patch("matplotlib.pyplot.savefig") as mock_save,
        mock.patch("os.makedirs") as mock_makedirs,
        mock.patch("matplotlib.pyplot.show") as mock_show,
    ):
        modified_graph.create_graph()
        modified_graph.visualize_graph(
            modified_graph.graph, show=True, save=False
        )
        mock_show.assert_called_once()
        mock_save.assert_not_called()
        mock_makedirs.assert_not_called()


def test_visualize_partition_save(modified_graph):
    with (
        mock.patch("matplotlib.pyplot.savefig") as mock_save,
        mock.patch("os.makedirs") as mock_makedirs,
        mock.patch("matplotlib.pyplot.show") as mock_show,
    ):
        modified_graph.create_graph()
        modified_graph.apply_louvain()
        modified_graph.visualize_partition(
            modified_graph.louvain_subgraphs,
            show=False,
            save=True,
            title="test",
        )
        base_dir = os.path.join("exports/8_0-0_0-0-0-0-0_0-1-0-0-0/images")
        mock_makedirs.assert_called_with(base_dir, exist_ok=True)
        name = "/test.png"
        mock_save.assert_called_with(base_dir + name, dpi=300)
        mock_show.assert_not_called()


def test_visualize_partition_show(modified_graph):
    with (
        mock.patch("matplotlib.pyplot.savefig") as mock_save,
        mock.patch("os.makedirs") as mock_makedirs,
        mock.patch("matplotlib.pyplot.show") as mock_show,
    ):
        modified_graph.create_graph()
        modified_graph.apply_louvain()
        modified_graph.visualize_partition(
            modified_graph.louvain_subgraphs, show=True, save=False
        )
        mock_show.assert_called_once()
        mock_save.assert_not_called()
        mock_makedirs.assert_not_called()
