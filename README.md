## Similarity-based Graph Generator

Generation and processing of a graph from pairwise similarity measures.

## Installation

To install similarity_graph_generator from GitHub repository, do:

```console
git clone https://github.com/https://github.com/matchms/similarity-graph-generator.git
cd similarity-graph-generator
python -m pip install .
# or on mac: python3 -m pip install .
```

## Documentation

This Python library allows the user to apply various modifications and community detection algorithms on a provided matrix of similarity measures and a graph created from the matrix. `main.py` shows the examplary use of the Python library. 

On the provided matrix a threshold and a normalization of the values can be applied. The values can be visualized as a histogram. After that a graph can be created. Different thresholds can also be applied on the graph and the edge weights can then be normalized. Different community detection algorithms can be applied to the graph and the results can be visualized in multiple ways. The images can then be exported or the graph and the recovered community partitions can be exportd as graphml files in order to visulize them with other tools.