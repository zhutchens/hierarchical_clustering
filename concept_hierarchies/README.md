# Ontology hierarchy construction
This folder contains a method of generating ontological hierarchies.

The method consists of the following:
## preprocessing.py
Preprocessing and filtering functions based on `tf`, `tf_idf` and `nltk` for part-of-speech tagging. The final output of these functions is a dataframe of filtered-out words (options for exclusion of stopwords, verbs, determiners, pronouns, adverbs and numbers) together with their `tf`, `df` and `tf_idf` measures. The dataframe will be ordered with respect to the desired measure (default is `tf_idf`). From this dataframe, one can extract the `top_n_words` with respect to a desired measure and pass them onto the relation extractor.

## extract_relations.py
Given the filtered-out `top_n_words` and a list of all sentences of a corpus of text, the function `get_directed_relations` will extract directed relations between the words. The `top_n_words` do not need to come from the preprocessing functions. They can be chosen arbitrarily. 

The relations are found using SpaCy dependency tree parsing by extracting subject-object pairs, identifying negative relations, replacing pronouns, using conjunct subjects and so on. As a simple example, from the sentence 

"Then Jesus sent the multitude away, and went into the house." 

the relation extractor will extract two relations: the relation (Jesus->multitude) and, using a conjunct subject, the relation (Jesus->house). But from the following sentence

"No man hath seen God at any time."

the inverted relation (God->man) will be extracted due to the negative determiner attached to the word "man". One can say, that the relation extractor searches for action-oriented relations.

Before feeding the relations into the iterative hierarchy construction algorithm, the relations need to be ordered. To do this, the `order_directed_relations` function can be used with a number of options. As the hierarchy construction algorithm is iterative, the order of the relations given to it has a big impact on the outcome. It is advised to try out multiple settings combinations, there is not a one-fits-all solution. For more settings details, refer to the documentation of the respective functions.

## ontology_algorithm.py
A simple iterative hierarchy construction algorithm `construct_ontology_hierarchy`. Upon construction, the hierarchy is constantly checked for cycles. Additionally, there is a function `print_hierarchy_tree_from_ontology` which can print the hierarchy out: 

<img src="ontology_hierarchy_example.png" alt= “” width="30%" height="30%">

Newer visualisation options have been implemented. Notably the `draw_hierarchy_tree_from_ontology`.

## topic_modeling.py
Contains the k-means clustering algorithm for topic modeling. Calling `kmeans_tfidf_clustering` with chapters and the number of desired topics will return that many clusters of chapters given.

## integrative_workflow.py
Integrates the topic modelling and relation extraction modules. Main function `construct_topic_modeling_concept_hierarchy` generates a concept hierarchy given chapters. It calls the topic modeling module to obtain the chapter clusters and key terms per cluster. One can select a cluster for which to generate a concept hierarchy using either key terms of even manually added terms as roots.

## Application examples
Application of the whole workflow is demonstrated in Jupyter notebooks. For some we use the KJV Bible as the corpus of text and for others the Theology Reconsidered corpus:

- `genesis_ontology_new.py`: generation of a ontological hierarchy of key terms in Genesis. One can specify the `last_chapter` that will be included in the analysis and `n` as the number of key terms taken from the preprocessing.
- `whole_bible_ontology.ipynb`: generation of an ontological hierarchy of key terms in any set of books of the Bible. The set of books can be specified in the `chosen_books` variable and the number of key terms extraced in the `n` variable.
- `key_term_tree_bible.ipynb`: generation of an ontological hierarchy using the integrated workflow. The simplest and easiest way of using the workflow with quite some freedom of specifying algorithm details. Crucial function is the `construct_topic_modeling_concept_hierarchy` with the most important inputs `num_topics` and `chosen_cluster`.
- `theology_reconsidered_ontology.ipynb`: same integrated workflow applied to the Theology Reconsidered corpus.