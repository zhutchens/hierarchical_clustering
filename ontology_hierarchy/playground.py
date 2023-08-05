import re

from extract_relations import get_directed_relations, order_directed_relations
from ontology_algorithm import (
    construct_ontology_hierarchy,
    draw_hierarchy_tree_from_ontology,
    print_hierarchy_tree_from_ontology,
)
from preprocessing import (
    get_gospel_top_70_words_dictionary,
    get_word_types_with_tf_idf,
    preprocess_kjv,
)
from topic_modeling import filter_topic_modeling_key_terms, kmeans_tfidf_clustering

# kjv_path = "/Users/zebo/Documents/Freelancing/upwork/Peter_J_Worth_Jr/NLP/hierarchical_clustering/data/t_kjv.csv"
# kjv_bible_df = preprocess_kjv(
#     path_to_kjv=kjv_path,
# )

# # Get book column unique values.
# all_books = kjv_bible_df["book"].unique()

# chosen_books = [
#     "Matthew",
#     "Luke",
#     # "Acts",
#     # "Revelation",
#     "John",
#     "Mark",
# ]


# print("Chosen books: ", chosen_books)

# # Specify the number of top words to use.
# n = 40

# text_per_chapter = []
# for book in chosen_books:
#     book_df = kjv_bible_df[kjv_bible_df["book"] == book]
#     for chapter in book_df["chapter"].unique():
#         chapter_df = book_df[book_df["chapter"] == chapter]
#         text_per_chapter.append(" ".join(chapter_df["text"].values))
# print("Starting tf_idf_pre_filtering...")
# tf_idf_pre_filtering = get_word_types_with_tf_idf(
#     text_per_chapter,
#     "tf",
#     skip_stopwords=True,
#     include_verbs=False,
#     include_determiners=False,
#     include_pronouns=False,
#     include_adverbs=False,
#     include_numbers=False,
# )
# # Show firt n words in the tf_idf_pre_filtering dataframe.
# print(tf_idf_pre_filtering.head(n))
# breakpoint()
# top_n_words = tf_idf_pre_filtering.head(n)["word"].values

# # Create a list of all verses of the chosen books.
# all_verses = []
# for book in chosen_books:
#     book_df = kjv_bible_df[kjv_bible_df["book"] == book]
#     for chapter in book_df["chapter"].unique():
#         chapter_df = book_df[book_df["chapter"] == chapter]
#         for verse in chapter_df["text"].values:
#             all_verses.append(verse)

# top_n_words_gender_dict = get_gospel_top_70_words_dictionary()

# directed_relations, relations_to_verbs = get_directed_relations(
#     top_n_words=top_n_words,
#     all_verses=all_verses,
#     verbose=False,
#     top_n_words_gender_dictionary=top_n_words_gender_dict,
#     only_compounds=True,
#     get_all_one_directional=True,
# )
# breakpoint()

# ordered_directed_relations = order_directed_relations(
#     directed_relations=directed_relations,
#     tf_idf_pre_filtering=tf_idf_pre_filtering,
#     order_by="product",
#     include_ordering_wrt_occurences=True,
#     verbose=False,
# )


theology_reconsidered_path = "/Users/zebo/Documents/Freelancing/upwork/Peter_J_Worth_Jr/NLP/hierarchical_clustering/data/theology_reconsidered.txt"

with open(theology_reconsidered_path, "r") as f:
    theology_reconsidered = f.read()


def split_into_chapters(input_filename, verbose=False):
    """Split the input file into chapters."""
    titles = []
    chapters = []
    # Open the input file and read its contents
    with open(input_filename, "r", encoding="utf-8") as input_file:
        contents = input_file.read()

    # Split the contents into chunks based on the separator criteria
    chunks = contents.split("\n\n\n\n")  # empty lines in the text

    ch_cnt = 0

    # Create an output file for each chunk
    for i, chunk in enumerate(chunks):
        # Skip empty chunks
        if not chunk.strip():
            continue

        # Extract the title and content from the chunk
        lines = chunk.strip().split("\n")
        title = lines[0]
        content = "\n".join(lines[1:])

        if verbose:
            print("processing CH" + str(ch_cnt) + ": " + title + "...")

        # Create the output file
        chapters.append(content)
        titles.append(title)

        ch_cnt += 1
    return titles, chapters


titles, chapters = split_into_chapters(theology_reconsidered_path)

chapters[0]

# Should I remove the word Figure?

# Remove \n from the chapters.
chapters = [chapter.replace("\n", "") for chapter in chapters]
# Remove \t from the chapters.
chapters = [chapter.replace("\t", " ") for chapter in chapters]


# import re

# Separate the chapters into sentences.
sentences_per_chapter_prep = [re.split("\.  ", chapter) for chapter in chapters]
sentences_per_chapter_prep = [
    [sentence + "." for sentence in chapter] for chapter in sentences_per_chapter_prep
]

sentences_per_chapter = []
for chapter_prep in sentences_per_chapter_prep:
    chapter = []
    for sentence in chapter_prep:
        chapter.extend(re.split("(?<!i.e|etc|e.g|iii)\. ", sentence))
    sentences_per_chapter.append(chapter)

# sentences_per_chapter[0]
for title_index, title in enumerate(titles):
    print(title_index, title)


# Remove chapters, titles and sentences if the title starts with "Part ".
title_indices_to_remove = [
    title_index for title_index, title in enumerate(titles) if title.startswith("Part ")
]

titles = [
    title
    for title_index, title in enumerate(titles)
    if title_index not in title_indices_to_remove
]
sentences_per_chapter = [
    chapter
    for chapter_index, chapter in enumerate(sentences_per_chapter)
    if chapter_index not in title_indices_to_remove
]
chapters = [
    chapter
    for chapter_index, chapter in enumerate(chapters)
    if chapter_index not in title_indices_to_remove
]

NUM_TOPICS = 8
clusters, key_terms_per_cluster = kmeans_tfidf_clustering(
    chapters=chapters,
    num_topics=NUM_TOPICS,
)

# # Specify the chapters used in the analysis.
# chosen_cluster = 7
# use_key_terms = True

# chosen_chapters = clusters[chosen_cluster]

# chapter_titles = [titles[i] for i in chosen_chapters]
# key_terms = key_terms_per_cluster[chosen_cluster]

# print("Chosen chapters ", chosen_chapters)

# # Specify the number of top words to use.
# n = max(50, len(chosen_chapters) * 5)

# text_per_chapter = []
# for chapter_idx in chosen_chapters:
#     text_per_chapter.append(chapters[chapter_idx])

# tf_idf_pre_filtering = get_word_types_with_tf_idf(
#     text_per_chapter,
#     "tf",
#     skip_stopwords=True,
#     include_verbs=False,
#     include_determiners=False,
#     include_pronouns=False,
#     include_adverbs=False,
#     include_numbers=False,
# )

# if use_key_terms:
#     top_n_words = filter_topic_modeling_key_terms(
#         key_terms=key_terms_per_cluster[chosen_cluster],
#         tf_idf_word_types=tf_idf_pre_filtering,
#         verbose=True,
#     )
#     # top_n_words = ["dàodé jīng"]
# else:
#     top_n_words = tf_idf_pre_filtering.head(n)["word"].values

# # See difference between top words and key terms.
# # print("Top words: ", top_n_words)
# # print("Key terms: ", key_terms)
# # print("Overlap: ", set(top_n_words).intersection(set(key_terms)))
# # print("Symmetric difference: ", set(top_n_words).symmetric_difference(set(key_terms)))


# # Create a list of all verses of the chosen books.
# all_verses = []
# for chapter_idx in chosen_chapters:
#     all_verses.extend(sentences_per_chapter[chapter_idx])

# # Extract the directed relations.
# directed_relations, relations_to_verbs = get_directed_relations(
#     top_n_words=top_n_words,
#     all_verses=all_verses,
#     only_compounds=True,
#     get_all_one_directional="lower",
#     verbose=True,
# )
# breakpoint()

# # Order the directed relations.
# ordered_directed_relations = order_directed_relations(
#     directed_relations=directed_relations,
#     tf_idf_pre_filtering=tf_idf_pre_filtering,
#     order_by="tf_idf",
#     include_ordering_wrt_occurences=True,
#     verbose=True,
# )
# breakpoint()

# # Construct the ontology hierarchy.
# ontology_hierarchy, words_with_parents = construct_ontology_hierarchy(
#     ordered_directed_relations=ordered_directed_relations,
# )

# # print_hierarchy_tree_from_ontology(
# #     ontological_hierarchy=ontology_hierarchy,
# #     words_with_parents=words_with_parents,
# # )
# draw_hierarchy_tree_from_ontology(
#     ontological_hierarchy=ontology_hierarchy,
#     relations_to_verbs=relations_to_verbs,
#     title="Theology reconsidered topic modeling cluster "
#     + str(chosen_cluster)
#     + " with "
#     + str(len(chosen_chapters))
#     + " chapters",
#     topic_modelling_chapters=chapter_titles,
# )

chosen_cluster = 7
use_key_terms = True
chosen_hierarchy_depth = 5
chosen_hierarchy_max_width = 5
only_compounds = True

chosen_chapters = clusters[chosen_cluster]

chapter_titles = [titles[i] for i in chosen_chapters]
key_terms = key_terms_per_cluster[chosen_cluster]

print("Chosen chapters ", chosen_chapters)

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

if use_key_terms:
    current_level_words = filter_topic_modeling_key_terms(
        key_terms=key_terms_per_cluster[chosen_cluster],
        tf_idf_word_types=tf_idf_pre_filtering,
        verbose=True,
    )
    # current_level_words.append("dàodé jīng") #TODO: Remove this.
else:
    raise NotImplementedError("Not implemented yet.")
    # Specify the number of top words to use.
    n = max(50, len(chosen_chapters) * 5)
    current_level_words = tf_idf_pre_filtering.head(n)["word"].values

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
    only_compounds=only_compounds,
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
        print(
            "Removing ",
            relation[1],
            " from current_level_words, it will be a child of ",
            relation[0],
        )


for current_level in range(chosen_hierarchy_depth):
    print("\n Current level: ", current_level)

    if len(current_level_words) == 0:
        break

    # Now lets see all relations that come out of the key terms.
    directed_relations, relations_to_verbs = get_directed_relations(
        top_n_words=current_level_words,
        all_verses=all_verses,
        verbose=False,
        get_all_one_directional="lower",
        only_compounds=only_compounds,
    )
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
                word in tf_idf_pre_filtering["word"].values for word in child.split(" ")
            )
        )
    ]
    print("Children to remove due to tf_idf_pre_filtering: ", children_to_remove_1)

    children_to_remove_2 = [
        key[1] for key, _ in directed_relations.items() if key[1] in used_words
    ]
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

    # Constrain at most chosen_hierarchy_max_width relations per key term.
    if chosen_hierarchy_max_width is not None:
        counter = {}
        relations_to_remove = []
        for idx, relation in enumerate(ordered_directed_relations):
            if relation[0] in counter:
                counter[relation[0]] += 1
            else:
                counter[relation[0]] = 1
            if counter[relation[0]] > 5:
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
    print("Current level words ", current_level_words)


# Construct the ontology hierarchy.
ontology_hierarchy, words_with_parents = construct_ontology_hierarchy(
    ordered_directed_relations=all_ordered_directed_relations,
)

# Draw the ontology hierarchy.
draw_hierarchy_tree_from_ontology(
    ontological_hierarchy=ontology_hierarchy,
    relations_to_verbs=all_relations_to_verbs,
    title="Theology reconsidered topic modeling cluster "
    + str(chosen_cluster)
    + " with "
    + str(len(chosen_chapters))
    + " chapters",
    topic_modelling_chapters=chapter_titles,
)
