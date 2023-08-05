from typing import List

from extract_relations import get_directed_relations, order_directed_relations
from ontology_algorithm import construct_ontology_hierarchy
from preprocessing import get_word_types_with_tf_idf
from topic_modeling import filter_topic_modeling_key_terms, kmeans_tfidf_clustering


def construct_topic_modeling_concept_hierarchy(
    chapters: List,
    titles: List,
    sentences_per_chapter: List,
    num_topics: List,
    chosen_cluster: int = 0,
    hierarchy_depth: int = 5,
    hierarchy_max_width: int = None,
    n_key_terms: int = 10,
    manually_added_roots: List[str] = None,
    verbose: bool = False,
    # only_compounds: bool = True,
):
    """Generate a concept hierarchy from the given chapters using topic modeling.

    Generates a concept hierarchy from the given chapters. Topic modeling is used
    to cluster the chapters into topics and to extract key terms for each topic.
    The key terms are used as the roots of the concept hierarchy. Then the tree is
    iteratively constructed by finding the children of the roots and then the children
    of the children and so on. The children are found by extracting the relations
    between the current level words and the rest of the words in the chapters.

    Parameters
    ----------
    chapters : List
        A list of chapters. Each chapter is a string.
    titles : List
        A list of titles of the chapters.
    sentences_per_chapter : List
        A list of lists of sentences. Each sentence is a string.
    num_topics : int
        The number of topics to cluster the chapters into.
    chosen_cluster : int, optional
        The index of the cluster for which to generate the concept hierarchy. The default is 0.
    hierarchy_depth : int, optional
        The depth of the concept hierarchy. The default is 5.
    hierarchy_max_width : int, optional
        The maximum number of children per node. The default is None meaning no limit.
    n_key_terms : int, optional
        The number of key terms to extract for each cluster. The default is 10.
    manually_added_roots : List[str], optional
        A list of manually added roots. The default is None.
    verbose : bool, optional
        Whether to print the progress. The default is False.

    Returns
    -------
    ontology_hierarchy : Dict
        The generated ontology hierarchy.
    all_relations_to_verbs : Dict
        A dictionary to keep track of verbs of the relations.
    chapter_titles : List
        A list of the titles of the chosen chapters.
    """
    clusters, key_terms_per_cluster = kmeans_tfidf_clustering(
        chapters=chapters,
        num_topics=num_topics,
        n_key_terms=n_key_terms,
    )

    chosen_chapters = clusters[chosen_cluster]

    chapter_titles = [titles[i] for i in chosen_chapters]
    if verbose:
        print("\n Chosen chapters ", chosen_chapters)

    text_per_chapter = []
    for chapter_idx in chosen_chapters:
        text_per_chapter.append(chapters[chapter_idx])

    tf_idf_pre_filtering = get_word_types_with_tf_idf(
        text_per_chapter,
        "tf",
        skip_stopwords=True,
        include_verbs=False,
        include_determiners=False,
        include_pronouns=False,
        include_adverbs=False,
        include_numbers=True,
    )

    current_level_words = filter_topic_modeling_key_terms(
        key_terms=key_terms_per_cluster[chosen_cluster],
        tf_idf_word_types=tf_idf_pre_filtering,
        verbose=True,
    )
    if manually_added_roots is not None:
        current_level_words.extend(manually_added_roots)

    # Create a list of all verses of the chosen books.
    all_verses = []
    for chapter_idx in chosen_chapters:
        all_verses.extend(sentences_per_chapter[chapter_idx])

    used_words = set()
    all_ordered_directed_relations = []
    all_relations_to_verbs = {}

    # First, let's see relations between the key terms.
    directed_relations, relations_to_verbs = get_directed_relations(
        top_n_words=current_level_words,
        all_verses=all_verses,
        verbose=False,
        only_compounds=True,
    )

    # Order the directed relations.
    ordered_directed_relations = order_directed_relations(
        directed_relations=directed_relations,
        tf_idf_pre_filtering=tf_idf_pre_filtering,
        order_by="product",
        include_ordering_wrt_occurences=True,
        verbose=False,
    )

    for relation in ordered_directed_relations:
        if relation[0] in current_level_words and relation[1] in current_level_words:
            current_level_words.remove(relation[1])
            if verbose:
                print(
                    "Removing ",
                    relation[1],
                    " from current_level_words, it will be a child of ",
                    relation[0],
                )

    for current_level in range(hierarchy_depth):
        if verbose:
            print("\n Current level: ", current_level)

        if len(current_level_words) == 0:
            break

        # Now lets see all relations that come out of the key terms.
        directed_relations, relations_to_verbs = get_directed_relations(
            top_n_words=current_level_words,
            all_verses=all_verses,
            verbose=False,
            get_all_one_directional="lower",
            only_compounds=True,
        )
        if verbose:
            print("All children: ", [relation[1] for relation in directed_relations])

        # Filter out the relations whose children is not in tf_idf_pre_filtering
        # and it has not been used already, but print them out first.
        children_to_remove_1 = [
            key[1]
            for key, _ in directed_relations.items()
            if key[1] not in tf_idf_pre_filtering["word"].values
        ]
        # Keep the bigrams and trigrams which are not in tf_idf_pre_filtering but
        # all of their words are in tf_idf_pre_filtering.
        children_to_remove_1 = [
            child
            for child in children_to_remove_1
            if not (
                len(child.split(" ")) > 1
                and all(
                    word in tf_idf_pre_filtering["word"].values
                    for word in child.split(" ")
                )
            )
        ]
        if verbose:
            print(
                "Children to remove due to tf_idf_pre_filtering: ", children_to_remove_1
            )

        children_to_remove_2 = [
            key[1] for key, _ in directed_relations.items() if key[1] in used_words
        ]
        if verbose:
            print("Children to remove due to used words: ", children_to_remove_2)

        directed_relations = {
            key: value
            for key, value in directed_relations.items()
            if key[1] not in children_to_remove_1 and key[1] not in children_to_remove_2
        }

        # Order the directed relations.
        ordered_directed_relations = order_directed_relations(
            directed_relations=directed_relations,
            tf_idf_pre_filtering=tf_idf_pre_filtering,
            order_by="product",
            include_ordering_wrt_occurences=True,
            verbose=False,
        )

        # Constrain at most hierarchy_max_width relations per key term.
        if hierarchy_max_width is not None:
            counter = {}
            relations_to_remove = []
            for _, relation in enumerate(ordered_directed_relations):
                if relation[0] in counter:
                    counter[relation[0]] += 1
                else:
                    counter[relation[0]] = 1
                if counter[relation[0]] > hierarchy_max_width:
                    relations_to_remove.append(relation)
            for relation in relations_to_remove:
                ordered_directed_relations.remove(relation)

        # Add parents and children to the used words.
        used_words.update([relation[0] for relation in ordered_directed_relations])
        used_words.update([relation[1] for relation in ordered_directed_relations])

        # Add the relations to the list of all relations.
        all_ordered_directed_relations.extend(ordered_directed_relations)
        all_relations_to_verbs.update(relations_to_verbs)

        # Set the current level words to the words that are the children of the current level words.
        current_level_words = [relation[1] for relation in ordered_directed_relations]
        if verbose:
            print("Current level words: ", current_level_words)

    # Construct the ontology hierarchy.
    ontology_hierarchy, _ = construct_ontology_hierarchy(
        ordered_directed_relations=all_ordered_directed_relations,
    )

    return ontology_hierarchy, all_relations_to_verbs, chapter_titles
