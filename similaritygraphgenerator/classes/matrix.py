import copy

import numpy as np
import textdistance


class Matrix:
    def __init__(self, compounds_list):
        self.options = {
            "noise": False,
            "noise_level": False,
            "noise_exponent": False,
        }
        self.compounds_list = compounds_list
        self.__create_similarity_matrix()

    def __create_similarity_matrix(self):
        """
        Create similarity matrix for compounds in compounds_list.

        Square matrix where each element at (i, j) represents the similarity
        score between the sequences of the compounds c1 and c2. Each element
        is calculated using __overlap_fraction function, which compares the
        sequences of two compounds. The diagonal of the matrix is set to 0.0
        to avoid self-loops later in the graph.
        """
        num_compounds = len(self.compounds_list)
        similarities = np.zeros((num_compounds, num_compounds))
        for i, c1 in enumerate(self.compounds_list):
            for j, c2 in enumerate(self.compounds_list):
                similarities[i, j] = self.__overlap_fraction(
                    "".join(c1.sequence), "".join(c2.sequence)
                )
        np.fill_diagonal(similarities, 0.0)
        self.similarity_matrix = similarities
        if not hasattr(self, "original_similarity_matrix"):
            self.original_similarity_matrix = copy.deepcopy(similarities)

    def __overlap_fraction(self, seq1, seq2):
        """
        Calculate the similarity of two provided sequences using longest common
        substring method. The identified overlap of both sequences is then
        normalized to represent the similarity of the two sequences.

        Args:
            seq1 (String): The first sequence for comparison
            seq2 (String): The second sequence for comparison

        Returns:
            similarity: Normalized similarity of the two sequences.
                        1 indicating sequences are identical,
                        0 indicating they have no common substring.
        """
        overlap = len(textdistance.lcsstr(seq1, seq2))
        similarity = 2 * overlap / (len(seq1) + len(seq2))
        return similarity

    def add_noise_to_matrix(self, percentage_to_modify=10, noise_level=0.1):
        """
        Add random noise to similarity matrix.

        Args:
            percentage_to_modify (int): Percentage of upper triangular elements
                                        excluding the diagonal to add noise to.
                                        Defaults to 10.
            noise_level (float): Scaling factor for noise values.
                                Defaults to 0.1.
        """
        upper_tri_indices = np.triu_indices_from(self.similarity_matrix, k=1)
        total_elements = upper_tri_indices[0].size
        num_to_modify = int(
            round((percentage_to_modify / 100) * total_elements)
        )
        selected_indices = np.random.choice(
            total_elements, size=num_to_modify, replace=False
        )

        noise_values = np.random.uniform(low=-1, high=1, size=num_to_modify)
        noise_values = (
            np.sign(noise_values) * np.abs(noise_values) ** 3 * noise_level
        )

        noise_matrix = np.zeros_like(self.similarity_matrix)
        i, j = upper_tri_indices
        noise_matrix[i[selected_indices], j[selected_indices]] = noise_values

        noise_matrix += noise_matrix.T
        noisy_matrix = self.similarity_matrix + noise_matrix
        np.fill_diagonal(noisy_matrix, 0)

        self.options["noise"] = percentage_to_modify
        self.options["noise_level"] = noise_level
        self.options["noise_exponent"] = 3
        self.similarity_matrix = noisy_matrix
