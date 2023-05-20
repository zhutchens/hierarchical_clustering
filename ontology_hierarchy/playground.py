from extract_relations import get_directed_relations, order_directed_relations
from ontology_algorithm import (
    construct_ontology_hierarchy,
    print_hierarchy_tree_from_ontology,
)
from preprocessing import (
    get_gospel_top_70_words_dictionary,
    get_word_types_with_tf_idf,
    preprocess_kjv,
)

kjv_path = "/Users/zebo/Documents/Freelancing/upwork/Peter_J_Worth_Jr/NLP/hierarchical_clustering/data/t_kjv.csv"
kjv_bible_df = preprocess_kjv(
    path_to_kjv=kjv_path,
)

# Get book column unique values.
all_books = kjv_bible_df["book"].unique()

chosen_books = [
    "Matthew",
    "Luke",
    # "Acts",
    # "Revelation",
    "John",
    "Mark",
]


print("Chosen books: ", chosen_books)

# Specify the number of top words to use.
n = 70

text_per_chapter = []
for book in chosen_books:
    book_df = kjv_bible_df[kjv_bible_df["book"] == book]
    for chapter in book_df["chapter"].unique():
        chapter_df = book_df[book_df["chapter"] == chapter]
        text_per_chapter.append(" ".join(chapter_df["text"].values))

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

top_n_words = tf_idf_pre_filtering.head(n)["word"].values

# Create a list of all verses of the chosen books.
all_verses = []
for book in chosen_books:
    book_df = kjv_bible_df[kjv_bible_df["book"] == book]
    for chapter in book_df["chapter"].unique():
        chapter_df = book_df[book_df["chapter"] == chapter]
        for verse in chapter_df["text"].values:
            all_verses.append(verse)

top_n_words_gender_dict = get_gospel_top_70_words_dictionary()

directed_relations = get_directed_relations(
    top_n_words=top_n_words,
    all_verses=all_verses,
    verbose=True,
    top_n_words_gender_dictionary=top_n_words_gender_dict,
)

ordered_directed_relations = order_directed_relations(
    directed_relations=directed_relations,
    tf_idf_pre_filtering=tf_idf_pre_filtering,
    order_by="product",
    include_ordering_wrt_occurences=True,
    verbose=True,
)
# ordered_directed_relations
