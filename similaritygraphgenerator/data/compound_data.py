type_rules1 = {
    "name": "Type1",
    "start": "AAAA",
    "additions": (2, 6),
    "blocks": ["A", "C", "D"],
}

type_rules2 = {
    "name": "Type2",
    "start": "AACCC",
    "additions": (3, 6),
    "blocks": ["B", "E"],
}

type_rules3 = {
    "name": "Type3",
    "description": "low variability class, \
        should lead to relatively low inner-class similarities",
    "start": "CCCCB",
    "additions": (1, 4),
    "blocks": ["C", "D", "E"],
}

type_rules4 = {
    "name": "Type4",
    "description": "low variability class, \
        should lead to relatively low inner-class similarities, \
            overlaps with Type3",
    "start": "ECCCCA",
    "additions": (1, 4),
    "blocks": ["C", "D", "E"],
}

type_rules5 = {
    "name": "Type5",
    "description": "potential overlaps with Type 1",
    "start": "CCAAA",
    "additions": (2, 6),
    "blocks": ["A", "C", "E"],
}

type_rules6 = {
    "name": "Type6",
    "description": "high variability class, \
        should lead to relatively low inner-class similarities",
    "start": "DDDD",
    "additions": (7, 10),
    "blocks": ["C", "D", "E"],
}

type_rules7 = {
    "name": "Type7",
    "description": "high variability class, overlap with Type6 ",
    "start": "AADD",
    "additions": (6, 9),
    "blocks": ["A", "D", "E"],
}

type_rules8 = {
    "name": "Type8",
    "description": "class with high inner-class similarities",
    "start": "ABCDE",
    "additions": (1, 10),
    "blocks": ["A"],
}

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

gn_recipe_good = [
    (32, type_rules1),
    (32, type_rules2),
    (32, type_rules5),
    (32, type_rules8),
]

gn_recipe_not_good = [
    (32, type_rules3),
    (32, type_rules4),
    (32, type_rules6),
    (32, type_rules7),
]

good_recipe = [
    (96, type_rules1),
    (96, type_rules2),
    (32, type_rules3),
    (32, type_rules4),
    (96, type_rules5),
    (32, type_rules6),
    (32, type_rules7),
    (96, type_rules8),
]

bad_recipe = [
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

big_good_recipe = [
    (480, type_rules1),
    (480, type_rules2),
    (160, type_rules3),
    (160, type_rules4),
    (480, type_rules5),
    (160, type_rules6),
    (160, type_rules7),
    (480, type_rules8),
]

big_bad_recipe = [
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