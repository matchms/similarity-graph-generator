import pytest
from similaritygraphgenerator.classes.compound_generator import Compound
from similaritygraphgenerator.classes.matrix import Matrix


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
def original_matrix(some_compounds_list):
    matrix = Matrix(some_compounds_list)
    return matrix


@pytest.fixture
def modified_matrix(some_compounds_list):
    matrix = Matrix(some_compounds_list)
    return matrix


def test_add_no_noise_to_matrix(original_matrix, modified_matrix):
    modified_matrix.add_noise_to_matrix(percentage_to_modify=0)
    assert (
        original_matrix.similarity_matrix == modified_matrix.similarity_matrix
    ).all()


def test_add_noise_to_matrix(original_matrix, modified_matrix):
    modified_matrix.add_noise_to_matrix()
    assert (
        original_matrix.similarity_matrix == modified_matrix.similarity_matrix
    ).any()
