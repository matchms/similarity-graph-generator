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
