# Ontology hierarchy construction
This folder contains a method of generating ontological hierarchies.

The method consists of the following:
- `preprocessing.py` : preprocessing and filtering functions based on `tf`, `tf_idf` and `nltk` for part-of-speech tagging. The final output of these functions is a dataframe of filtered words (options for exclusion of stopwords, verbs, determiners, pronouns, adverbs and numbers) with `tf`, `df` and `tf_idf` measures. The dataframe will ordered with respect to the desired measure (default is `tf_idf`). From this dataframe, one can extract the `top_n_words` with respect to a desired measure and pass onto the relation extractor.

- `extract_relations.py` : given the filtered `top_n_words` and a list of all sentences of the corpus of text, the function `get_directed_relations` will extract directed relations between the words. The relations are found using SpaCy dependency tree parsing by extracting subject-object pairs, identifying negative relations, replacing pronouns, using conjunct subjects and so on. As a simple example, from the sentence 

"Then Jesus sent the multitude away, and went into the house." 

the relation extractor will extract two relations: (Jesus->multitude) and using a conjunct subject (Jesus->house). But from the following sentence

"No man hath seen God at any time."

the opposite relation (God->man) will be extracted due to the negative determiner attached to the word "man". One can say, that the relation extractor searches for action-oriented relations.

Before feeding the relations into the iterative hierarchy construction algorithm, the relations need to be ordered. To do this, the `order_directed_relations` function can be used with a number of options. As the hierarchy construction algorithm is iterative, the order of the relations given to it has a big impact on the outcome.

- `ontology_algorithm.py`: a simple iterative hierarchy construction algorithm `construct_ontology_hierarchy` and a function which prints the hierarchy out. Upon construction, the hierarchy is constantly checked for cycles.
