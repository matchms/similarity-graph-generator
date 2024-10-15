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
