import numpy as np


class DummyCompound:
    """Simple dummy compound class."""

    def __init__(self, sequence_arr, compound_type):
        self.sequence_arr = sequence_arr
        self.sequence = "".join(self.sequence_arr)
        self.compound_type = compound_type

    def __repr__(self):
        return "Sequence: " + self.sequence


class DummyCompoundGenerator:
    """Generator for dummy compounds based on rules sets in the `recipe`."""

    def __init__(self, recipe, seed=0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.recipe = recipe
        self.blocks = ["A", "B", "C", "D", "E"]

    def generate_compound(self, type_rules):
        start_sequence = np.array(list(type_rules["start"]), dtype="<U1")
        num_additions = self.rng.integers(*type_rules["additions"])  # Choose a random length within the specified range
        additions = self.rng.choice(
            type_rules["blocks"], size=num_additions, replace=True
        )  # Generate additional blocks
        sequence = np.concatenate((start_sequence, additions))  # Combine the start sequence with the generated bl
        return DummyCompound(sequence, type_rules["name"])

    def generate_compound_list(self):
        compounds = []
        for num, rules in self.recipe:
            for _ in range(num):
                compounds.append(self.generate_compound(rules))
        return compounds
