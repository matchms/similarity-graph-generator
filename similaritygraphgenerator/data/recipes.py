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

recipe = [
    (50, type_rules1),
    (50, type_rules2),
    (50, type_rules3),
    (50, type_rules4),
    (50, type_rules5),
    (20, type_rules6),
    (20, type_rules7),
    (50, type_rules8),
]

small_well_defined_recipe = [
    (32, type_rules1),
    (32, type_rules2),
    (32, type_rules5),
    (32, type_rules8),
]

small_weak_recipe = [
    (32, type_rules3),
    (32, type_rules4),
    (32, type_rules6),
    (32, type_rules7),
]

well_defined_recipe = [
    (96, type_rules1),
    (96, type_rules2),
    (32, type_rules3),
    (32, type_rules4),
    (96, type_rules5),
    (32, type_rules6),
    (32, type_rules7),
    (96, type_rules8),
]

weak_recipe = [
    (32, type_rules1),
    (32, type_rules2),
    (96, type_rules3),
    (96, type_rules4),
    (32, type_rules5),
    (96, type_rules6),
    (96, type_rules7),
    (32, type_rules8),
]

equal_recipe = [
    (64, type_rules1),
    (64, type_rules2),
    (64, type_rules3),
    (64, type_rules4),
    (64, type_rules5),
    (64, type_rules6),
    (64, type_rules7),
    (64, type_rules8),
]

random_recipe = [
    (54, type_rules1),
    (40, type_rules2),
    (97, type_rules3),
    (23, type_rules4),
    (40, type_rules5),
    (84, type_rules6),
    (49, type_rules7),
    (125, type_rules8),
]

big_well_defined_recipe = [
    (480, type_rules1),
    (480, type_rules2),
    (160, type_rules3),
    (160, type_rules4),
    (480, type_rules5),
    (160, type_rules6),
    (160, type_rules7),
    (480, type_rules8),
]

big_weak_recipe = [
    (160, type_rules1),
    (160, type_rules2),
    (480, type_rules3),
    (480, type_rules4),
    (160, type_rules5),
    (480, type_rules6),
    (480, type_rules7),
    (160, type_rules8),
]

big_equal_recipe = [
    (320, type_rules1),
    (320, type_rules2),
    (320, type_rules3),
    (320, type_rules4),
    (320, type_rules5),
    (320, type_rules6),
    (320, type_rules7),
    (320, type_rules8),
]

big_random_recipe = [
    (1012, type_rules1),
    (213, type_rules2),
    (52, type_rules3),
    (193, type_rules4),
    (831, type_rules5),
    (66, type_rules6),
    (148, type_rules7),
    (45, type_rules8),
]
