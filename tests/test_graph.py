import os
from unittest import mock

import pytest
from similaritygraphgenerator.classes.compound_generator import Compound
from similaritygraphgenerator.classes.graph import Graph
from similaritygraphgenerator.data.compound_data import (
    type_rules1,
    type_rules2,
    type_rules3,
    type_rules4,
    type_rules5,
    type_rules6,
    type_rules7,
    type_rules8,
)


@pytest.fixture
def some_recipe():
    recipe = [
        (1, type_rules1),
        (1, type_rules2),
        (1, type_rules3),
        (1, type_rules4),
        (1, type_rules5),
        (1, type_rules6),
        (1, type_rules7),
        (1, type_rules8),
    ]
    return recipe


@pytest.fixture
def some_compounds_list():
    compounds_list = [
        Compound(["A" "A" "A" "A" "C" "C" "A" "A" "A"], "Type1"),
        Compound(["A" "A" "C" "C" "C" "B" "B" "E"], "Type2"),
        Compound(["C" "C" "C" "C" "B" "E" "D"], "Type3"),
        Compound(["E" "C" "C" "C" "C" "A" "E" "E"], "Type4"),
        Compound(["C" "C" "A" "A" "A" "C" "C" "E" "A"], "Type5"),
        Compound(["D" "D" "D" "D" "E" "C" "D" "E" "D" "C"], "Type6"),
        Compound(["A" "A" "D" "D" "A" "E" "A" "D" "A" "A"], "Type7"),
        Compound(["A" "B" "C" "D" "E" "A" "A" "A" "A" "A"], "Type8"),
    ]
    return compounds_list


@pytest.fixture
def original_graph(some_recipe, some_compounds_list):
    graph = Graph(some_recipe, some_compounds_list)
    return graph


@pytest.fixture
def modified_graph(some_recipe, some_compounds_list):
    graph = Graph(some_recipe, some_compounds_list)
    return graph


def test_add_no_noise_to_matrix(original_graph, modified_graph):
    modified_graph.add_noise_to_matrix(percentage_to_modify=0)
    assert (
        original_graph.similarity_matrix == modified_graph.similarity_matrix
    ).all()


def test_add_noise_to_matrix(original_graph, modified_graph):
    modified_graph.add_noise_to_matrix()
    assert (
        original_graph.similarity_matrix == modified_graph.similarity_matrix
    ).any()


def test_apply_no_matrix_treshold(original_graph, modified_graph):
    modified_graph.apply_matrix_treshold(percentage_to_remove=0)
    assert (
        original_graph.similarity_matrix == modified_graph.similarity_matrix
    ).all()


def test_apply_matrix_treshold(original_graph, modified_graph):
    modified_graph.apply_matrix_treshold()
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


def test_add_no_false_edges(original_graph, modified_graph):
    original_graph.create_graph()
    modified_graph.create_graph()
    modified_graph.add_false_edges()
    assert original_graph.graph.edges() == modified_graph.graph.edges()


def test_add_false_edges(original_graph, modified_graph):
    original_graph.create_graph()
    modified_graph.create_graph()
    modified_graph.apply_edge_threshold_global(percentage_to_remove=100)
    modified_graph.add_false_edges()
    assert not original_graph.graph.edges() == modified_graph.graph.edges()


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
        base_dir = os.path.join(
            "exports/8-8_0-0-0-0-0_0-0-0-0-0-0_0-0-0-0-0/images/"
        )
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
        base_dir = os.path.join(
            "exports/8-8_0-0-0-0-0_0-0-0-0-0-0_0-0-0-0-0/images"
        )
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
        modified_graph.visualize_partition(
            modified_graph.original_subgraphs,
            show=False,
            save=True,
            title="test",
        )
        base_dir = os.path.join(
            "exports/8-8_0-0-0-0-0_0-0-0-0-0-0_0-0-0-0-0/images"
        )
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
        modified_graph.visualize_partition(
            modified_graph.original_subgraphs, show=True, save=False
        )
        mock_show.assert_called_once()
        mock_save.assert_not_called()
        mock_makedirs.assert_not_called()


def test_calculate_scores(modified_graph):
    modified_graph.create_graph()
    modified_graph.apply_louvain()
    modified_graph.calculate_scores()
    assert modified_graph.original_score.homogeneity == 1
    assert modified_graph.original_score.ari == 1
    assert modified_graph.original_score.nmi == 1
    assert modified_graph.louvain_scores.homogeneity > 0
    assert modified_graph.louvain_scores.modularity > 0
    assert modified_graph.louvain_scores.nmi > 0
