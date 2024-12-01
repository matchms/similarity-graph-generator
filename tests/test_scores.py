from types import SimpleNamespace

import pytest
from similaritygraphgenerator.classes.compound_generator import Compound
from similaritygraphgenerator.classes.graph import Graph
from similaritygraphgenerator.classes.scores import Scores
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
def some_graph(some_recipe, some_compounds_list):
    graph = Graph(some_recipe, some_compounds_list)
    graph.create_graph()
    return graph


@pytest.fixture
def some_scores(some_graph):
    score = Scores(
        "graph",
        some_graph.graph,
        some_graph.original_subgraphs,
        some_graph.original_partition_list,
    )
    score.name = "Test"
    score.homogeneity = 0.85
    score.completeness = 0.90
    score.community_size = SimpleNamespace(
        score=0.75, num_com=1, avg_com_size=1
    )
    score.modularity = 0.65
    score.ari = 0.78
    score.nmi = 0.82
    score.ami = 0.80
    score.fmi = 0.76
    return score


def test_str(some_scores):
    expected_output = (
        "+++ SCORES FOR TEST +++\n"
        "+++ Quality metrics +++\n"
        "Homogeneity: 0.85\n"
        "Completeness: 0.90\n"
        "Community Size: 0.75 with 1.00 communities and 1.00 avg community size\n"
        "Modularity: 0.65\n"
        "+++ Clustering metrics +++\n"
        "Adjusted Rand Index: 0.78\n"
        "Normalized Mutual Information: 0.82\n"
        "Adjusted Mutual Information: 0.80\n"
        "Fowlkes Mallows Index: 0.76\n"
    )
    assert str(some_scores) == expected_output
