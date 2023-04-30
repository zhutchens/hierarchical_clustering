import spacy
import pandas as pd

from preprocessing import (
    get_top_30_gender_number_dictionary,
)

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

    directed_relations = set()

    subject_object_in_top_n_words = 0

    noun_pos = ["NOUN", "PROPN"]

    pronouns = ["i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its", "we", "us", "our", "ours", "they", "them", "their", "theirs"]

    for verse_idx, verse in enumerate(all_verses):

        doc = nlp(verse)
        doc_sents = [s for s in doc.sents]
        if verbose:
            print("\n", len(doc_sents), " sentences in verse ", verse_idx)
        for sent in doc_sents:
            if verbose:
                print("sentence: ", sent.text)
                print("ents: ", sent.ents)

            sentence_verbs = [token for token in sent if token.pos_ in ["VERB", "AUX"] and token.i]
            sentence_subjects_non_pronouns = [
                token for token in sent if token.dep_ in ["nsubj", "nsubjpass"] 
                and token.text.lower() not in pronouns 
                and token.text.lower() in top_n_words_gender_dictionary.keys()
            ]
            for root in sentence_verbs:
                conjunct_subject = None
                # If there is no subject in the sentence, check if there is a subject in a conjunct
                # and use that as the subject.
                if len([child for child in root.children if child.dep_ in ["nsubj", "nsubjpass"]]) == 0 \
                and len([child for child in root.children if child.dep_ in ["dobj", "pobj"]]) > 0:
                    if root.dep_ == "conj":
                        root_head = root.head
                        for child in root_head.children:
                            if child.dep_ in ["nsubj", "nsubjpass"]:
                                conjunct_subject = child
                                break
                    if conjunct_subject is None:
                        continue 
                # NOT USING, TOO STRONG, MAYBE USE LATER WITH NUMBER OF RELATIONS TODO
                # HMMMMM no, actually it seems fine. There was another bug...

                # Check that there is a subject and an object.
                if len([child for child in root.children if child.dep_ in ["nsubj", "nsubjpass"]]) == 0 \
                or len([child for child in root.children if child.dep_ in ["dobj", "pobj"]]) == 0:
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
            
                #If there is a conjunct subject, use that as the subject. NOT USING, SEE UP TODO
                if conjunct_subject is not None and current_subject is None:
                    if verbose:
                        print("Using conjunct subject: ", conjunct_subject.text)
                    current_subject = conjunct_subject
                    subject_text = conjunct_subject.text

                # Check if subject is a pronoun.
                # If it is, replace with a non-pronoun subject
                # from the sentence.
                if current_subject.text.lower() in pronouns:
                    child_gender = current_subject.morph.get('Gender')[0] if len(current_subject.morph.get('Gender'))>0 else "Neut"
                    child_number = current_subject.morph.get('Number')[0] if len(current_subject.morph.get('Number'))>0 else None
                    # Take the subjects from sentence_subjects_non_pronouns that occured before the pronoun
                    # and order them opposite to the order of the sentence.
                    filtered_subjects_for_pronoun = [
                        subject for subject in sentence_subjects_non_pronouns
                        if subject.i < current_subject.i
                    ][::-1]
                    subjects_with_same_gender_and_number = [subject for subject in filtered_subjects_for_pronoun 
                                                          if child_gender==top_n_words_gender_dictionary[subject.text.lower()][0] 
                                                          and child_number==top_n_words_gender_dictionary[subject.text.lower()][1]]
                    if len(subjects_with_same_gender_and_number) > 0:
                        old_subject = current_subject
                        current_subject = subjects_with_same_gender_and_number[0]
                        if verbose:
                            print("pronoun ", old_subject.text, "replaced with subject: ", current_subject.text)
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
                                if grandchild.text.lower() in top_n_words and subject_text.lower() in top_n_words and grandchild.text.lower() != subject_text.lower():
                                #if subject.pos_ in noun_pos and grandchild.pos_ in noun_pos:
                                    #print((subject.lower(), grandchild.text.lower()))
                                    directed_relations.add((subject_text.lower(), grandchild.text.lower()))
                                subject_object_in_top_n_words += int(grandchild.text.lower() in top_n_words and subject_text.lower() in top_n_words)

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
                                if grandchild.text.lower() in top_n_words and subject_text.lower() in top_n_words and grandchild.text.lower() != subject_text.lower():
                                #if subject.pos_ in noun_pos and grandchild.pos_ in noun_pos:
                                    #print((subject.lower(), grandchild.text.lower()))
                                    directed_relations.add((subject_text.lower(), grandchild.text.lower()))
                                subject_object_in_top_n_words += int(grandchild.text.lower() in top_n_words and subject_text.lower() in top_n_words)

                    # Search for direct objects through the dobj and pobj dependencies.
                    if child.dep_ in ["dobj", "pobj"]:
                        if verbose:
                            print("object: ", child.text)
                        if child.text in objects:
                            objects[child.text] += 1
                        else:
                            objects[child.text] = 1
                        if child.text.lower() in top_n_words and subject_text.lower() in top_n_words and child.text.lower() != subject_text.lower():
                        #if subject.pos_ in noun_pos and child.pos_ in noun_pos: 
                            #print((subject.lower(), child.text.lower()))
                            directed_relations.add((subject_text.lower(), child.text.lower()))
                        subject_object_in_top_n_words += int(child.text.lower() in top_n_words and subject_text.lower() in top_n_words)

                        # Check if the object is in conjunction with another object.
                        for grandchild in child.children:
                            if grandchild.dep_ in ["conj"]:
                                #print("object: ", grandchild.text)
                                if grandchild.text in objects:
                                    objects[grandchild.text] += 1
                                else:
                                    objects[grandchild.text] = 1
                                if grandchild.text.lower() in top_n_words and subject_text.lower() in top_n_words and grandchild.text.lower() != subject_text.lower():
                                #if subject.pos_ in noun_pos and grandchild.pos_ in noun_pos: 
                                    #print((subject.lower(), grandchild.text.lower()))
                                    directed_relations.add((subject_text.lower(), grandchild.text.lower()))
                                subject_object_in_top_n_words += int(grandchild.text.lower() in top_n_words and subject_text.lower() in top_n_words)
                            
                            # Continue searching for objects through the prep dependency.
                            if grandchild.dep_ in ["prep"]:
                                for great_grandchild in grandchild.children:
                                    if great_grandchild.dep_ in ["pobj"]:
                                        if verbose:
                                            print("object: ", great_grandchild.text)
                                        if great_grandchild.text in objects:
                                            objects[great_grandchild.text] += 1
                                        else:
                                            objects[great_grandchild.text] = 1
                                        if great_grandchild.text.lower() in top_n_words and subject_text.lower() in top_n_words and great_grandchild.text.lower() != subject_text.lower():
                                        #if subject.pos_ in noun_pos and great_grandchild.pos_ in noun_pos: 
                                            #print((subject.lower(), great_grandchild.text.lower()))
                                            directed_relations.add((subject_text.lower(), great_grandchild.text.lower()))
                                        subject_object_in_top_n_words += int(great_grandchild.text.lower() in top_n_words and subject_text.lower() in top_n_words)
                    

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
        print(subject_object_in_top_n_words, " of the subjects and objects were in the top 30 words.")

    return directed_relations

def order_directed_relations(
    directed_relations: set,
    tf_idf_pre_filtering: pd.DataFrame,
):
    """Order the directed relations with respect to the number of relations and the tf_idf of the first word of the relation.
    
    Parameters
    ----------
    directed_relations : set
        Set of directed relations.
    tf_idf_pre_filtering : pd.DataFrame
        Dataframe with columns words and tf_idf.
    
    Returns
    -------
    ordered_directed_relations : list
        List of ordered directed relations.
    """
    ordered_directed_relations = list(directed_relations)

    first_words = list(set([relation[0] for relation in ordered_directed_relations]))

    # Get the number of relations for each word in which it's superior.
    number_of_relations = {}
    for word in first_words:
        number_of_relations[word] = len([relation[0] for relation in ordered_directed_relations if relation[0]==word]) 

    # Get tf_idf from the dataframe tf_idf_pre_filtering with columns words and tf_idf.
    tf_idf_of_words = {}
    for word in first_words:
        tf_idf_of_words[word] = tf_idf_pre_filtering[tf_idf_pre_filtering["word"]==word]["tf_idf"].values[0]
        
    # Order the first words with respect to the number of relations and the tf_idf.
    first_words.sort(key=lambda x: number_of_relations[x]*tf_idf_of_words[x], reverse=True)


    # Order the relations with respect to the first words.
    ordered_directed_relations.sort(key=lambda x: first_words.index(x[0]), reverse=False)

    # If there are two relations with same words, keep only the one first in the list.
    for (first, second) in ordered_directed_relations:
        if (second, first) in ordered_directed_relations and \
            ordered_directed_relations.index((second, first)) > ordered_directed_relations.index((first, second)):
            ordered_directed_relations.remove((second, first))

    first_words.sort(key=lambda x: tf_idf_of_words[x], reverse=True)
    ordered_directed_relations.sort(key=lambda x: first_words.index(x[0]), reverse=False)

    return ordered_directed_relations


