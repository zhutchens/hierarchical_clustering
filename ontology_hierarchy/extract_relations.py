import pandas as pd
import spacy
from preprocessing import get_top_30_gender_number_dictionary


def get_directed_relations(
    top_n_words,
    all_verses,
    top_n_words_gender_dictionary: dict = None,
    verbose=False,
):
    """Extracts directed relations of given words from given corpus.

    Generates directed relations between the top_n_words given from the
    all_verses of a corpus. The directed relations are extracted by
    parsing sentence dependency trees from spaCy and looking for Subject - object
    relations. The relations are stored in a set of tuples. The set is returned.

    Parameters
    ----------
    top_n_words : list
        List of top n words for which to extract relations.
    all_verses : list
        List of verses to extract relations from.

    Returns
    -------
    set
        Set of tuples of directed relations.
    """
    nlp = spacy.load("en_core_web_lg")
    subjects = {}
    objects = {}

    if top_n_words_gender_dictionary is None:
        top_n_words_gender_dictionary = get_top_30_gender_number_dictionary()

    directed_relations = {}

    n_extracted_relations = 0

    pronouns = [
        "i",
        "me",
        "my",
        "mine",
        "you",
        "your",
        "yours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "we",
        "us",
        "our",
        "ours",
        "they",
        "them",
        "their",
        "theirs",
    ]

    negative_words = [
        "no",
        "not",
        "none",
        "nobody",
        "nothing",
        "neither",
        "nowhere",
        "never",
        "hardly",
        "scarcely",
        "barely",
    ]

    for verse_idx, verse in enumerate(all_verses):
        doc = nlp(verse)
        doc_sents = [s for s in doc.sents]
        if verbose:
            print("\n", len(doc_sents), " sentences in verse ", verse_idx)
        if verse_idx == 640:
            breakpoint()
        for sent in doc_sents:
            if verbose:
                print("sentence: ", sent.text)
                print("ents: ", sent.ents)

            sentence_verbs = [
                token for token in sent if token.pos_ in ["VERB", "AUX"] and token.i
            ]
            sentence_subjects_non_pronouns = [
                token
                for token in sent
                if token.dep_ in ["nsubj", "nsubjpass"]
                and token.text.lower() not in pronouns
                and token.text.lower() in top_n_words_gender_dictionary.keys()
            ]
            for root in sentence_verbs:
                conjunct_subject = None
                # If there is no subject in the sentence, check if there is a subject in a conjunct
                # and use that as the subject.
                if (
                    len(
                        [
                            child
                            for child in root.children
                            if child.dep_ in ["nsubj", "nsubjpass"]
                        ]
                    )
                    == 0
                ):
                    if root.dep_ == "conj":
                        root_head = root.head
                        for child in root_head.children:
                            if child.dep_ in ["nsubj", "nsubjpass"]:
                                conjunct_subject = child
                                break
                    if conjunct_subject is None:
                        continue

                current_subject = None
                # Iterate over the verb children to find the subject.
                for child in root.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        if verbose:
                            print("subject: ", child.text)
                        if child.text in subjects:
                            subjects[child.text] += 1
                        else:
                            subjects[child.text] = 1
                        current_subject = child

                if conjunct_subject is not None and current_subject is None:
                    if verbose:
                        print("Using conjunct subject: ", conjunct_subject.text)
                    current_subject = conjunct_subject
                    subject_text = conjunct_subject.text

                subject_negative_determiner = False
                # Look for negative determiner dependencies in the subject.
                for child in current_subject.children:
                    if child.dep_ in ["det"] and child.text.lower() in negative_words:
                        if verbose:
                            print("Negative subject determiner: ", child.text, ".")
                        subject_negative_determiner = True

                verb_negative_adverb = False
                # Look for negative adverb dependencies in the verb.
                for child in root.children:
                    if (
                        child.dep_ in ["neg", "advmod"]
                        and child.text.lower() in negative_words
                    ):
                        if verbose:
                            print("Negative adverb: ", child.text, ".")
                        verb_negative_adverb = True

                # Check if subject is a pronoun.
                # If it is, replace with a non-pronoun subject
                # from the sentence.
                if current_subject.text.lower() in pronouns:
                    child_gender = (
                        current_subject.morph.get("Gender")[0]
                        if len(current_subject.morph.get("Gender")) > 0
                        else "Neut"
                    )
                    child_number = (
                        current_subject.morph.get("Number")[0]
                        if len(current_subject.morph.get("Number")) > 0
                        else None
                    )
                    # Take the subjects from sentence_subjects_non_pronouns that occured before the pronoun
                    # and order them opposite to the order of the sentence.
                    filtered_subjects_for_pronoun = [
                        subject
                        for subject in sentence_subjects_non_pronouns
                        if subject.i < current_subject.i
                    ][::-1]
                    subjects_with_same_gender_and_number = [
                        subject
                        for subject in filtered_subjects_for_pronoun
                        if child_gender
                        == top_n_words_gender_dictionary[subject.text.lower()][0]
                        and child_number
                        == top_n_words_gender_dictionary[subject.text.lower()][1]
                    ]
                    if len(subjects_with_same_gender_and_number) > 0:
                        old_subject = current_subject
                        current_subject = subjects_with_same_gender_and_number[0]
                        if verbose:
                            print(
                                "pronoun ",
                                old_subject.text,
                                "replaced with subject: ",
                                current_subject.text,
                            )
                subject_text = current_subject.text

                # Start searching for objects.
                for child in root.children:
                    # Search for indirect objects through the dative dependency.
                    if child.dep_ in ["dative"]:
                        if verbose:
                            print("dative: ", child.text)
                        for grandchild in child.children:
                            if grandchild.dep_ in ["pobj"]:
                                if verbose:
                                    print("indirect object: ", grandchild.text)
                                if grandchild.text in objects:
                                    objects[grandchild.text] += 1
                                else:
                                    objects[grandchild.text] = 1
                                if (
                                    grandchild.text.lower() in top_n_words
                                    and subject_text.lower() in top_n_words
                                    and grandchild.text.lower() != subject_text.lower()
                                ):
                                    object_negative_determiner = (
                                        search_for_object_negative_determiner(
                                            grandchild, negative_words, verbose=verbose
                                        )
                                    )
                                    add_directed_relation(
                                        directed_relations,
                                        subject_text.lower(),
                                        grandchild.text.lower(),
                                        subject_negative_determiner,
                                        verb_negative_adverb,
                                        object_negative_determiner,
                                        verbose=verbose,
                                    )
                                    n_extracted_relations += 1

                    # Search for indirect objects through the prep dependency.
                    if child.dep_ in ["prep"]:
                        for grandchild in child.children:
                            if grandchild.dep_ in ["pobj"]:
                                if verbose:
                                    print("object: ", grandchild.text)
                                if grandchild.text in objects:
                                    objects[grandchild.text] += 1
                                else:
                                    objects[grandchild.text] = 1
                                if (
                                    grandchild.text.lower() in top_n_words
                                    and subject_text.lower() in top_n_words
                                    and grandchild.text.lower() != subject_text.lower()
                                ):
                                    object_negative_determiner = (
                                        search_for_object_negative_determiner(
                                            grandchild, negative_words, verbose=verbose
                                        )
                                    )
                                    add_directed_relation(
                                        directed_relations,
                                        subject_text.lower(),
                                        grandchild.text.lower(),
                                        subject_negative_determiner,
                                        verb_negative_adverb,
                                        object_negative_determiner,
                                        verbose=verbose,
                                    )
                                    n_extracted_relations += 1
                                # Continue searching for objects through conj and prep dependencies.
                                n_extracted_relations = search_objects_through_conj_prep_dependencies(
                                    grandchild,
                                    objects,
                                    top_n_words,
                                    subject_text,
                                    negative_words,
                                    directed_relations,
                                    subject_negative_determiner,
                                    verb_negative_adverb,
                                    n_extracted_relations,
                                    verbose,
                                )
                                

                    # Search for direct objects through the dobj and pobj dependencies.
                    if child.dep_ in ["dobj", "pobj"]:
                        if verbose:
                            print("object: ", child.text)
                        if child.text in objects:
                            objects[child.text] += 1
                        else:
                            objects[child.text] = 1
                        if (
                            child.text.lower() in top_n_words
                            and subject_text.lower() in top_n_words
                            and child.text.lower() != subject_text.lower()
                        ):
                            object_negative_determiner = (
                                search_for_object_negative_determiner(
                                    child, negative_words, verbose=verbose
                                )
                            )
                            add_directed_relation(
                                directed_relations,
                                subject_text.lower(),
                                child.text.lower(),
                                subject_negative_determiner,
                                verb_negative_adverb,
                                object_negative_determiner,
                                verbose=verbose,
                            )
                            n_extracted_relations += 1

                        # Continue searching for objects through conj and prep dependencies.
                        n_extracted_relations = search_objects_through_conj_prep_dependencies(
                            child,
                            objects,
                            top_n_words,
                            subject_text,
                            negative_words,
                            directed_relations,
                            subject_negative_determiner,
                            verb_negative_adverb,
                            n_extracted_relations,
                            verbose,
                        )

    # Create a dataframe with columns words, was_subject, was_object.
    subjects_df = pd.DataFrame(
        {"word": list(subjects.keys()), "was_subject": list(subjects.values())}
    )
    objects_df = pd.DataFrame(
        {"word": list(objects.keys()), "was_object": list(objects.values())}
    )

    # Merge the two dataframes.
    subjects_objects_df = pd.merge(subjects_df, objects_df, on="word", how="outer")

    # Fill NaN values with 0.
    subjects_objects_df = subjects_objects_df.fillna(0)
    # TODO I don't return this dataframe, it was here mostly for debugging purposes.
    # Can be removed later.

    if verbose:
        print("Number of extracted relations: ", n_extracted_relations)

    return directed_relations


def search_objects_through_conj_prep_dependencies(
        object,
        objects,
        top_n_words,
        subject_text,
        negative_words,
        directed_relations,
        subject_negative_determiner,
        verb_negative_adverb,
        n_extracted_relations,
        verbose: bool=False,
):
    for child in object.children:
        if child.dep_ in ["conj"]:
            # print("object: ", grandchild.text)
            if child.text in objects:
                objects[child.text] += 1
            else:
                objects[child.text] = 1
            if (
                child.text.lower() in top_n_words
                and subject_text.lower() in top_n_words
                and child.text.lower() != subject_text.lower()
            ):
                object_negative_determiner = (
                    search_for_object_negative_determiner(
                        child, negative_words, verbose=verbose
                    )
                )
                add_directed_relation(
                    directed_relations,
                    subject_text.lower(),
                    child.text.lower(),
                    subject_negative_determiner,
                    verb_negative_adverb,
                    object_negative_determiner,
                    verbose=verbose,
                )
                n_extracted_relations += 1

        # Continue searching for objects through the prep dependency.
        if child.dep_ in ["prep"]:
            for grandchild in child.children:
                if grandchild.dep_ in ["pobj"]:
                    if verbose:
                        print("object: ", grandchild.text)
                    if grandchild.text in objects:
                        objects[grandchild.text] += 1
                    else:
                        objects[grandchild.text] = 1
                    if (
                        grandchild.text.lower() in top_n_words
                        and subject_text.lower() in top_n_words
                        and grandchild.text.lower()
                        != subject_text.lower()
                    ):
                        object_negative_determiner = (
                            search_for_object_negative_determiner(
                                grandchild,
                                negative_words,
                                verbose=verbose,
                            )
                        )
                        add_directed_relation(
                            directed_relations,
                            subject_text.lower(),
                            grandchild.text.lower(),
                            subject_negative_determiner,
                            verb_negative_adverb,
                            object_negative_determiner,
                            verbose=verbose,
                        )
                        n_extracted_relations += 1
    return n_extracted_relations


def search_for_object_negative_determiner(token, negative_words, verbose=False):
    """Search for negative determiner dependencies in the object.

    Parameters
    ----------
    token : spacy.tokens.token.Token
        The object token.

    Returns
    -------
    bool
        Whether the object has a negative determiner.
    """
    object_negative_determiner = False
    for child in token.children:
        if child.dep_ in ["det"] and child.text.lower() in negative_words:
            if verbose:
                print("Negative object determiner: ", child.text, ".")
            object_negative_determiner = True
    return object_negative_determiner


def add_directed_relation(
    directed_relations: dict,
    subject: str,
    object: str,
    subject_negative_determiner: bool = False,
    object_negative_determiner: bool = False,
    verb_negative_adverb: bool = False,
    verbose: bool = False,
):
    """Adds a directed relation to the set of directed relations.

    Parameters
    ----------
    directed_relations : dict
        Dictionary of directed relations. Key is the (subject, object) tuple, value is the number of relations.
    subject : str
        Subject of the relation.
    object : str
        Object of the relation.
    subject_negative_determiner : bool, optional
        Whether the subject has a negative determiner, by default False
    """
    revert_order = (
        subject_negative_determiner + verb_negative_adverb + object_negative_determiner
    ) % 2 == 1
    if revert_order:
        if (object, subject) in directed_relations:
            directed_relations[(object, subject)] += 1
        else:
            directed_relations[(object, subject)] = 1
        if verbose:
            print("Inverting order of relation.")
            print("Adding relation: '", object, "' -> '", subject, "'")
    else:
        if (subject, object) in directed_relations:
            directed_relations[(subject, object)] += 1
        else:
            directed_relations[(subject, object)] = 1
        if verbose:
            print("Adding relation: '", subject, "' -> '", object, "'")


def order_directed_relations(
    directed_relations: dict,
    tf_idf_pre_filtering: pd.DataFrame,
    order_by: str = "product",
    include_ordering_wrt_occurences: bool = True,
    verbose: bool = False,
):
    """Order the directed relations with respect to the number of relations and the tf_idf of the first word of the relation.

    Parameters
    ----------
    directed_relations : dict
        Dictionary of directed relations. Key is the (subject, object) tuple, value is the number of relations.
    tf_idf_pre_filtering : pd.DataFrame
        Dataframe with columns words and tf_idf.
    order_by : str, optional
        The metric to order the relations by. Can be "tf", "tf_idf", "number_of_relations" or "product." By default "tf_idf".
    include_ordering_wrt_occurences : bool, optional
        Whether to include the ordering with respect to the number of occurances of the relation. By default True.

    Returns
    -------
    ordered_directed_relations : list
        List of ordered directed relations.
    """
    ordered_directed_relations = list(directed_relations.keys())

    first_words = list(set([relation[0] for relation in ordered_directed_relations]))

    # Remove a relation if there's the opposite relation with more occurences.
    for first, second in ordered_directed_relations:
        if (
            second,
            first,
        ) in ordered_directed_relations and directed_relations[
            (second, first)
        ] > directed_relations[(first, second)]:
            ordered_directed_relations.remove((first, second))

    # Get the number of relations for each word in which it's superior.
    number_of_relations = {}
    for word in first_words:
        number_of_relations[word] = sum(
            [
                directed_relations[relation]
                for relation in ordered_directed_relations
                if relation[0] == word
            ]
        )

    # Get tf_idf from the dataframe tf_idf_pre_filtering with columns words and tf_idf.
    tf_idf_of_words = {}
    for word in first_words:
        tf_idf_of_words[word] = tf_idf_pre_filtering[
            tf_idf_pre_filtering["word"] == word
        ]["tf_idf"].values[0]

    # Get tf from the dataframe tf_idf_pre_filtering with columns words and tf.
    tf_of_words = {}
    for word in first_words:
        tf_of_words[word] = tf_idf_pre_filtering[tf_idf_pre_filtering["word"] == word][
            "tf"
        ].values[0]

    # Order the first words with respect to the number of relations and the tf_idf.
    if order_by == "tf_idf":
        first_words.sort(
            key=lambda x: number_of_relations[x] * tf_idf_of_words[x], reverse=True
        )
    else:
        first_words.sort(
            key=lambda x: number_of_relations[x] * tf_of_words[x], reverse=True
        )

    # Order the relations with respect to the first words.
    ordered_directed_relations.sort(
        key=lambda x: first_words.index(x[0]), reverse=False
    )

    # If there are two relations with same words, keep only the one first in the list.
    for first, second in ordered_directed_relations:
        if (
            second,
            first,
        ) in ordered_directed_relations and ordered_directed_relations.index(
            (second, first)
        ) > ordered_directed_relations.index(
            (first, second)
        ):
            ordered_directed_relations.remove((second, first))

    if order_by == "tf":
        first_words.sort(key=lambda x: tf_of_words[x], reverse=True)
    if order_by == "tf_idf":
        first_words.sort(key=lambda x: tf_idf_of_words[x], reverse=True)
    elif order_by == "number_of_relations":
        first_words.sort(key=lambda x: number_of_relations[x], reverse=True)
    elif order_by == "product":
        first_words.sort(
            key=lambda x: number_of_relations[x] * tf_of_words[x], reverse=True
        )

    # Order the relations with respect to the number of occurances of the relation first
    # so directed_relations[word],
    # and then with respect to the order of the first word in the first_words list.
    if include_ordering_wrt_occurences:
        ordered_directed_relations.sort(
            key=lambda x: (directed_relations[x], -first_words.index(x[0])),
            reverse=True,
        )

        if verbose:
            # Make a dataframe with relations and occurances and print it completely
            occurances_of_directed_relations = [
                directed_relations[relation] for relation in ordered_directed_relations
            ]
            df = pd.DataFrame(
                {
                    "relation": ordered_directed_relations,
                    "occurances": occurances_of_directed_relations,
                }
            )
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                print(df)
    else:
        ordered_directed_relations.sort(
            key=lambda x: first_words.index(x[0]), reverse=False
        )

    return ordered_directed_relations
