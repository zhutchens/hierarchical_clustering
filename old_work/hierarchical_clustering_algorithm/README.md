# Hierarachical clustering
The hierarchical clustering algorithm provides the most relevant centers of a list of vectors in n-dimensional space in a hierarchical fashion. Relevance is measured as to having the most vectors in its proximity spread, where a proximity spread looks for vectors in the neighbourhood of a vector, and then continues spreading from those vectors with decreasing neighbourhood in each step until convergence. When those are found, the algorithm is restarted to search for the most relevant centers of the previously found centers. This creates a hierarchical structure of centers (or heads as stated in the code), the children of a center (head) in the second level are centers (heads) in the first level. 

## Application to word embeddings

The motivation of creating a hierarcical clustering of this kind was to use it with word embedding spaces: lists of vectors in n-dimensional space representing words of a corpus and their similarity, i.e. a lower dimensional representation of a vocabulary. This clustering can then be applied to obtain a hierarchical structure of the vocabulary, creating centers (heads) of the word embedding, forming a hierarchical structure of those and thus getting the most relevant words of the word embedding, their relations, and with it the relevant words of the corpus of text the word embedding was trained on.

## Coding details

The folder currently consists of:
- `clustering_class.py` -> the file containing the definition of the hierarchical clustering class with all the methods and options of the clustering.
- `clustering_class_testing.ipynb` -> a notebook with an example of usage of the algorithm on the `glove-wiki-gigaword-300` pretrained word embedding. The class is completely general and can be used on any pretrained word embedding.

### Plans

- Adding a notebook to explain all the functionalities and reasons behind them.
- Writing a better documentation of the methods.