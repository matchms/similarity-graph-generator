import pytest
from similaritygraphgenerator.classes.compound_generator import (
    CompoundGenerator,
)
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


def test_generate_compound(some_recipe):
    generator = CompoundGenerator(some_recipe, seed=0)
    compounds_list = generator.generate_compound_list()
    assert len(compounds_list) == 8


def test_generate_compound_with_error():
    with pytest.raises(TypeError) as excinfo:
        generator = CompoundGenerator(recipe=None, seed=0)
        generator.generate_compound_list()
    assert str(excinfo.value) == "'NoneType' object is not iterable"
