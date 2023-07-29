from preprocessing import (
    preprocess_kjv,
    get_word_types_with_tf_idf,
    get_gospel_top_70_words_dictionary,
)
from extract_relations import (
    get_directed_relations,
    order_directed_relations,
)
from ontology_algorithm import (
    construct_ontology_hierarchy,
    print_hierarchy_tree_from_ontology,
    draw_hierarchy_tree_from_ontology,
)
from topic_modeling import (
    kmeans_tfidf_clustering,
    filter_topic_modeling_key_terms,
)
import re

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
    with open(input_filename, 'r', encoding='utf-8') as input_file:
        contents = input_file.read()

    # Split the contents into chunks based on the separator criteria
    chunks = contents.split('\n\n\n\n')             # empty lines in the text

    ch_cnt = 0

    # Create an output file for each chunk
    for i, chunk in enumerate(chunks):
        # Skip empty chunks
        if not chunk.strip():
            continue

        # Extract the title and content from the chunk
        lines = chunk.strip().split('\n')
        title = lines[0]
        content = '\n'.join(lines[1:])

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
sentences_per_chapter_prep = [[sentence + "." for sentence in chapter] for chapter in sentences_per_chapter_prep]

sentences_per_chapter = []
for chapter_prep in sentences_per_chapter_prep:
    chapter = []
    for sentence in chapter_prep:
        chapter.extend(re.split("(?<!i.e|etc|e.g|iii)\. ", sentence))
    sentences_per_chapter.append(chapter)

#sentences_per_chapter[0]
for title_index, title in enumerate(titles):
    print(title_index, title)


# Remove chapters, titles and sentences if the title starts with "Part ".
title_indices_to_remove = [title_index for title_index, title in enumerate(titles) if title.startswith("Part ")]

titles = [title for title_index, title in enumerate(titles) if title_index not in title_indices_to_remove]
sentences_per_chapter = [chapter for chapter_index, chapter in enumerate(sentences_per_chapter) if chapter_index not in title_indices_to_remove]
chapters = [chapter for chapter_index, chapter in enumerate(chapters) if chapter_index not in title_indices_to_remove]

NUM_TOPICS = 8
clusters, key_terms_per_cluster = kmeans_tfidf_clustering(
    chapters=chapters,
    num_topics=NUM_TOPICS,
)

# Specify the chapters used in the analysis.
chosen_cluster = 7
use_key_terms = True

chosen_chapters = clusters[chosen_cluster]

chapter_titles = [titles[i] for i in chosen_chapters]
key_terms = key_terms_per_cluster[chosen_cluster]

print("Chosen chapters ", chosen_chapters)

# Specify the number of top words to use.
n = max(50, len(chosen_chapters)*5)

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
    include_numbers=False,
)

if use_key_terms:
    top_n_words = filter_topic_modeling_key_terms(
        key_terms=key_terms_per_cluster[chosen_cluster],
        tf_idf_word_types=tf_idf_pre_filtering,
        verbose=True,
    )
    top_n_words = ["dàodé jīng"]
else:
    top_n_words = tf_idf_pre_filtering.head(n)["word"].values

# See difference between top words and key terms.
# print("Top words: ", top_n_words)
# print("Key terms: ", key_terms)
# print("Overlap: ", set(top_n_words).intersection(set(key_terms)))
# print("Symmetric difference: ", set(top_n_words).symmetric_difference(set(key_terms)))


# Create a list of all verses of the chosen books.
all_verses = []
for chapter_idx in chosen_chapters:
    all_verses.extend(sentences_per_chapter[chapter_idx])

# Extract the directed relations.
directed_relations, relations_to_verbs = get_directed_relations(
top_n_words=top_n_words,
all_verses=all_verses,
only_compounds=True,
get_all_one_directional='lower',
verbose=True,
)

# Order the directed relations.
ordered_directed_relations = order_directed_relations(
    directed_relations=directed_relations,
    tf_idf_pre_filtering=tf_idf_pre_filtering,
    order_by="tf_idf",
    include_ordering_wrt_occurences=True,
    verbose=True,
)
breakpoint()

# Construct the ontology hierarchy.
ontology_hierarchy, words_with_parents = construct_ontology_hierarchy(
    ordered_directed_relations=ordered_directed_relations,
)

# print_hierarchy_tree_from_ontology(
#     ontological_hierarchy=ontology_hierarchy,
#     words_with_parents=words_with_parents,
# )
draw_hierarchy_tree_from_ontology(
    ontological_hierarchy=ontology_hierarchy,
    relations_to_verbs=relations_to_verbs,
    title="Theology reconsidered topic modeling cluster " + str(chosen_cluster) + " with " + str(len(chosen_chapters)) + " chapters",
    topic_modelling_chapters=chapter_titles,
)
