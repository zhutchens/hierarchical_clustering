import math
import re
from collections import Counter
from typing import Optional

import nltk
import numpy as np
import pandas as pd


def preprocess_kjv(
    path_to_kjv: str,
    get_book: Optional[str] = None,
):
    """Preprocess the King James Version of the Bible.

    Parameters
    ----------
    path_to_kjv : str
        Path to the King James Version of the Bible.
    get_book : Optional[str]
        If specified, also returns the book of the Bible specified.

    Returns
    -------
    return_dictionary : dict
    """
    df = pd.read_csv(path_to_kjv, index_col=False)

    # Rename books
    df.b.replace(
        {
            1: "Genesis",
            2: "Exodus",
            3: "Leviticus",
            4: "Numbers",
            5: "Deuteronomy",
            6: "Joshua",
            7: "Judges",
            8: "Ruth",
            9: "1 Samuel (1 Kings)",
            10: "2 Samuel (2 Kings)",
            11: "1 Kings (3 Kings)",
            12: "2 Kings (4 Kings)",
            13: "1 Chronicles",
            14: "2 Chronicles",
            15: "Ezra",
            16: "Nehemiah",
            17: "Esther",
            18: "Job",
            19: "Psalms",
            20: "Proverbs",
            21: "Ecclesiastes",
            22: "Song of Solomon (Canticles)",
            23: "Isaiah",
            24: "Jeremiah",
            25: "Lamentations",
            26: "Ezekiel",
            27: "Daniel",
            28: "Hosea",
            29: "Joel",
            30: "Amos",
            31: "Obadiah",
            32: "Jonah",
            33: "Micah",
            34: "Nahum",
            35: "Habakkuk",
            36: "Zephaniah",
            37: "Haggai",
            38: "Zechariah",
            39: "Malachi",
            40: "Matthew",
            41: "Mark",
            42: "Luke",
            43: "John",
            44: "Acts",
            45: "Romans",
            46: "1 Corinthians",
            47: "2 Corinthians",
            48: "Galatians",
            49: "Ephesians",
            50: "Philippians",
            51: "Colossians",
            52: "1 Thessalonians",
            53: "2 Thessalonians",
            54: "1 Timothy",
            55: "2 Timothy",
            56: "Titus",
            57: "Philemon",
            58: "Hebrews",
            59: "James",
            60: "1 Peter",
            61: "2 Peter",
            62: "1 John",
            63: "2 John",
            64: "3 John",
            65: "Jude",
            66: "Revelation",
        },
        inplace=True,
    )

    # Rename columns
    df.columns = ["id", "book", "chapter", "verse", "text"]

    if get_book is not None:
        book = df[df.book == get_book]
        return df, book

    return df


# make a better get_tf using Counter from collections
def better_get_tf_for_documents(documents, sortby="tf", skip_stopwords: bool = False):
    """Get the term frequency for each word in a list of documents.

    Much faster version of the same function. Uses Counter from collections.

    Parameters
    ----------
    verses : list
        A list of verses.
    sortby : str, optional
        The column to sort by, by default 'tf'. Can be chosen from ['word', 'tc', 'tf'].
    skip_stopwords : bool, optional
        Whether to skip stopwords, by default False.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the term frequency for each word in the documents.
    """
    tf = pd.DataFrame(columns=["word", "tc", "tf"])

    if skip_stopwords:
        from nltk.corpus import stopwords

        stopwords_set = set(stopwords.words("english"))
    else:
        stopwords_set = set()

    # Combine all documents into one string
    all_documents = " ".join(documents)

    # Get n_words
    n_words = len(re.findall(r"\w+", all_documents))

    # Get word counts
    word_counts = Counter(re.findall(r"\w+", all_documents))

    for word in word_counts:
        word_lower = word.lower()
        if word_lower not in tf["word"].values and word_lower not in stopwords_set:
            row = {}
            row["word"] = [word_lower]
            row["tf"] = [word_counts[word] / n_words]
            row["tc"] = word_counts[word]
            row_df = pd.DataFrame(row)
            tf = pd.concat([tf, row_df], ignore_index=True)
        elif word_lower not in stopwords_set:
            tf.loc[tf.word.isin([word_lower]), "tf"] += word_counts[word] / n_words
            tf.loc[tf.word.isin([word_lower]), "tc"] += word_counts[word]

    # Include bigrams
    for document in documents:
        bigrams = Counter(nltk.bigrams(re.findall(r"\w+", document)))

        for bigram in bigrams:
            bigram_lower = " ".join([word.lower() for word in bigram])
            if (
                bigram_lower not in tf["word"].values
                and bigram_lower.split(" ")[0] not in stopwords_set
                and bigram_lower.split(" ")[1] not in stopwords_set
            ):
                row = {}
                row["word"] = [bigram_lower]
                row["tf"] = [bigrams[bigram] / n_words]
                row["tc"] = bigrams[bigram]
                row_df = pd.DataFrame(row)
                tf = pd.concat([tf, row_df], ignore_index=True)
            elif (
                bigram_lower.split(" ")[0] not in stopwords_set
                and bigram_lower.split(" ")[1] not in stopwords_set
            ):
                tf.loc[tf.word.isin([bigram_lower]), "tf"] += bigrams[bigram] / n_words
                tf.loc[tf.word.isin([bigram_lower]), "tc"] += bigrams[bigram]

    # Include trigrams
    for document in documents:
        trigrams = Counter(nltk.trigrams(re.findall(r"\w+", document)))

        for trigram in trigrams:
            trigram_lower = " ".join([word.lower() for word in trigram])
            if (
                trigram_lower not in tf["word"].values
                and trigram_lower.split(" ")[0] not in stopwords_set
                and trigram_lower.split(" ")[1] not in stopwords_set
                and trigram_lower.split(" ")[2] not in stopwords_set
            ):
                row = {}
                row["word"] = [trigram_lower]
                row["tf"] = [trigrams[trigram] / n_words]
                row["tc"] = trigrams[trigram]
                row_df = pd.DataFrame(row)
                tf = pd.concat([tf, row_df], ignore_index=True)
            elif (
                trigram_lower.split(" ")[0] not in stopwords_set
                and trigram_lower.split(" ")[1] not in stopwords_set
                and trigram_lower.split(" ")[2] not in stopwords_set
            ):
                tf.loc[tf.word.isin([trigram_lower]), "tf"] += (
                    trigrams[trigram] / n_words
                )
                tf.loc[tf.word.isin([trigram_lower]), "tc"] += trigrams[trigram]

    tf = tf.sort_values(by=sortby, ascending=False)
    tf = tf.reset_index(drop=True)
    return tf


def get_tf_for_documents(documents, sortby="tf", skip_stopwords: bool = False):
    """Get the term frequency for each word in a list of documents.

    Parameters
    ----------
    verses : list
        A list of verses.
    sortby : str, optional
        The column to sort by, by default 'tf'. Can be chosen from ['word', 'tc', 'tf'].
    skip_stopwords : bool, optional
        Whether to skip stopwords, by default False.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the term frequency for each word in the documents.
    """
    raise DeprecationWarning("Use better_get_tf_for_documents instead.")
    tf = pd.DataFrame(columns=["word", "tc", "tf"])

    if skip_stopwords:
        from nltk.corpus import stopwords

        stopwords_set = set(stopwords.words("english"))
    else:
        stopwords_set = set()

    # Get n_words
    n_words = 0
    for verse in documents:
        verse_words = re.findall(r"\w+", verse)
        n_words += len(verse_words)

    for verse in documents:
        verse_words = re.findall(r"\w+", verse)
        for word in verse_words:
            word = word.lower()
            if word not in tf["word"].values and word not in stopwords_set:
                row = {}
                row["word"] = [word]
                row["tf"] = [1 / n_words]
                row["tc"] = 1
                row_df = pd.DataFrame(row)
                tf = pd.concat([tf, row_df], ignore_index=True)
            elif word not in stopwords_set:
                tf.loc[tf.word.isin([word]), "tf"] += 1 / n_words
                tf.loc[tf.word.isin([word]), "tc"] += 1

    tf = tf.sort_values(by=sortby, ascending=False)
    tf = tf.reset_index(drop=True)
    return tf


def get_idf_for_documents(documents, sortby="idf", skip_stopwords: bool = False):
    """Get the inverse document frequency for each word in a list of documents.

    Parameters
    ----------
    documents : list
        A list of documents. A document is a string of text.
    sortby : str, optional
        The column to sort by, by default 'idf'. Can be chosen from ['word', 'dc', 'idf'].
    skip_stopwords : bool, optional
        Whether to skip stopwords, by default False.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the inverse document frequency for each word in the documents.
    """
    idf = pd.DataFrame(columns=["word", "dc", "idf"])

    N_documents = len(documents)

    if skip_stopwords:
        from nltk.corpus import stopwords

        stopwords_set = set(stopwords.words("english"))
    else:
        stopwords_set = set()

    # Get all_verse_words
    terms_per_document = []
    for document in documents:
        document_terms = set()
        document_words = re.findall(r"\w+", document)
        for i in range(len(document_words)):
            document_words[i] = document_words[i].lower()
        document_terms.update(document_words)

        document_bigrams = nltk.bigrams(re.findall(r"\w+", document))
        for bigram in document_bigrams:
            bigram = " ".join(bigram)
            bigram = bigram.lower()
            document_terms.add(bigram)
        document_terms.update(document_bigrams)

        document_trigrams = nltk.trigrams(re.findall(r"\w+", document))
        for trigram in document_trigrams:
            trigram = " ".join(trigram)
            trigram = trigram.lower()
            document_terms.add(trigram)
        document_terms.update(document_trigrams)

        terms_per_document.append(document_terms)

    for document in documents:
        document_words = re.findall(r"\w+", document)
        for word in document_words:
            word = word.lower()
            if word not in idf["word"].values and word not in stopwords_set:
                row = {}
                row["word"] = [word]
                row["idf"] = math.log(
                    N_documents
                    / len(
                        [
                            True
                            for document_terms in terms_per_document
                            if word in document_terms
                        ]
                    )
                )
                row["dc"] = len(
                    [
                        True
                        for document_terms in terms_per_document
                        if word in document_terms
                    ]
                )
                row_df = pd.DataFrame(row)
                idf = pd.concat([idf, row_df], ignore_index=True)

        document_bigrams = nltk.bigrams(re.findall(r"\w+", document))
        for bigram in document_bigrams:
            bigram = " ".join(bigram)
            bigram = bigram.lower()
            if (
                bigram not in idf["word"].values
                and bigram.split(" ")[0] not in stopwords_set
                and bigram.split(" ")[1] not in stopwords_set
            ):
                row = {}
                row["word"] = [bigram]
                row["idf"] = math.log(
                    N_documents
                    / len(
                        [
                            True
                            for document_terms in terms_per_document
                            if bigram in document_terms
                        ]
                    )
                )
                row["dc"] = len(
                    [
                        True
                        for document_terms in terms_per_document
                        if bigram in document_terms
                    ]
                )
                row_df = pd.DataFrame(row)
                idf = pd.concat([idf, row_df], ignore_index=True)

        document_trigrams = nltk.trigrams(re.findall(r"\w+", document))
        for trigram in document_trigrams:
            trigram = " ".join(trigram)
            trigram = trigram.lower()
            if (
                trigram not in idf["word"].values
                and trigram.split(" ")[0] not in stopwords_set
                and trigram.split(" ")[1] not in stopwords_set
                and trigram.split(" ")[2] not in stopwords_set
            ):
                row = {}
                row["word"] = [trigram]
                row["idf"] = math.log(
                    N_documents
                    / len(
                        [
                            True
                            for document_terms in terms_per_document
                            if trigram in document_terms
                        ]
                    )
                )
                row["dc"] = len(
                    [
                        True
                        for document_terms in terms_per_document
                        if trigram in document_terms
                    ]
                )
                row_df = pd.DataFrame(row)
                idf = pd.concat([idf, row_df], ignore_index=True)

    idf = idf.sort_values(by=sortby, ascending=False)
    idf = idf.reset_index(drop=True)
    return idf


def get_tf_idf_for_documents(documents, sort_by="tf", skip_stopwords: bool = False):
    """Get the tf-idf for each word in a list of documents.

    Parameters
    ----------
    documents : list
        A list of documents. A document is a string of text.
    sort_by : str, optional
        The column to sort by, by default 'tf'. Can be chosen from ['word', 'dc', 'idf', 'tf', 'tf_idf'].
    skip_stopwords : bool, optional
        Whether to skip stopwords, by default False.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the tf-idf for each word in the documents.
    """

    tf = better_get_tf_for_documents(
        documents, sortby="word", skip_stopwords=skip_stopwords
    )
    tf = tf.reset_index(drop=True)
    idf = get_idf_for_documents(documents, sortby="word", skip_stopwords=skip_stopwords)
    idf = idf.reset_index(drop=True)

    idf_column = {"dc": idf["dc"].values, "idf": idf["idf"].values}
    idf_df = pd.DataFrame(idf_column)
    tf_idf = pd.concat([tf, idf_df], axis=1)

    tf_idf_column = {"tf_idf": tf["tf"].values * idf["idf"].values}
    tf_idf_df = pd.DataFrame(tf_idf_column)
    tf_idf = pd.concat([tf_idf, tf_idf_df], axis=1)

    tf_idf = tf_idf.sort_values(by=sort_by, ascending=False)
    return tf_idf


def categorize_words(verses, skip_stopwords: bool = False):
    """Categorize all words given in the list of verses.

    Uses the nltk pos_tagger to categorize the words.

    Parameters
    ----------
    verses : list
        A list of verses. A verse is a string of text.
    skip_stopwords : bool, optional
        Whether to skip stopwords, by default False.

    Returns
    -------
    dict
        A dictionary with the words as keys and a list of word type categores of that word as values.
    """

    if skip_stopwords:
        from nltk.corpus import stopwords

        stopwords_set = set(stopwords.words("english"))
    else:
        stopwords_set = set()

    nnp_set = set()

    word_types = {}

    # Get word_types
    for verse in verses:
        verse_tokenized = nltk.tokenize.word_tokenize(verse)

        verse_pos_tags = nltk.pos_tag(verse_tokenized)

        # print(verse_pos_tags)
        for word, tag in verse_pos_tags:
            if tag in ["NNP", "NP", "NNS", "NNPS", "NN"]:
                nnp_set.add(
                    (
                        word,
                        tag,
                    )
                )

        verse_ne = nltk.ne_chunk(verse_pos_tags, binary=False)

        for verse_word in verse_ne:
            if type(verse_word) == tuple:
                try:
                    verse_word[0].lower()
                except:
                    print(verse_word[0])
                if verse_word[0].lower() not in stopwords_set:
                    if verse_word[0].lower() not in word_types:
                        word_types[verse_word[0].lower()] = {verse_word[1]: 1}
                    else:
                        if verse_word[1] not in word_types[verse_word[0].lower()]:
                            word_types[verse_word[0].lower()][verse_word[1]] = 1
                        else:
                            word_types[verse_word[0].lower()][verse_word[1]] += 1
            else:
                label = verse_word.label()
                while verse_word:
                    verse_word_pop = verse_word.pop()
                    if verse_word_pop[0].lower() not in word_types:
                        word_types[verse_word_pop[0].lower()] = {
                            verse_word_pop[1]: 1,
                            label: 1,
                        }
                    else:
                        if (
                            verse_word_pop[1]
                            not in word_types[verse_word_pop[0].lower()]
                        ):
                            word_types[verse_word_pop[0].lower()][verse_word_pop[1]] = 1
                        else:
                            word_types[verse_word_pop[0].lower()][
                                verse_word_pop[1]
                            ] += 1
                        if label not in word_types[verse_word_pop[0].lower()]:
                            word_types[verse_word_pop[0].lower()][label] = 1
                        else:
                            word_types[verse_word_pop[0].lower()][label] += 1

    return word_types


def get_word_types_with_tf_idf(
    verses,
    sortby="tf",
    skip_stopwords=False,
    include_verbs=True,
    include_determiners=True,
    include_pronouns=True,
    include_adverbs=True,
    include_numbers=True,
    verbose=False,
):
    """Get a dataframe of words with their tf-idf scores and word types.

    Parameters
    ----------
    verses : list
        A list of verses. A verse is a string of text.
    sortby : str, optional
        Sort the dataframe by 'tf' or 'tf-idf', by default 'tf'
    skip_stopwords : bool, optional
        Whether to skip stopwords, by default False
    include_verbs : bool, optional
        Whether to include verbs, by default True
    include_determiners : bool, optional
        Whether to include determiners, by default True
    include_pronouns : bool, optional
        Whether to include pronouns, by default True

    Returns
    -------
    pd.DataFrame
        A dataframe with the words as index, tf-idf scores as columns, and word types as a column.
    """
    tf_idf = get_tf_idf_for_documents(
        verses, sort_by=sortby, skip_stopwords=skip_stopwords
    )
    tf_idf = tf_idf.reset_index(drop=True)

    word_types = categorize_words(verses, skip_stopwords=skip_stopwords)

    word_type_list = []
    for word in tf_idf["word"].values:
        # If word is in word types (not bigram) just add the word type
        if word in word_types:
            word_type_list.append(word_types[word])
        # If the word is a bigram, then add word types of both words
        elif (
            " " in word
            and len(word.split(" ")) == 2
            and word.split(" ")[0] in word_types
            and word.split(" ")[1] in word_types
        ):
            word_type_list.append(
                {
                    key: word_types[word.split(" ")[0]].get(key, 0)
                    + word_types[word.split(" ")[1]].get(key, 0)
                    for key in word_types[word.split(" ")[0]].keys()
                    | word_types[word.split(" ")[1]].keys()
                }
            )
        # If the word is a trigram, then add word types of all three words
        elif (
            " " in word
            and len(word.split(" ")) == 3
            and word.split(" ")[0] in word_types
            and word.split(" ")[1] in word_types
            and word.split(" ")[2] in word_types
        ):
            word_type_list.append(
                {
                    key: word_types[word.split(" ")[0]].get(key, 0)
                    + word_types[word.split(" ")[1]].get(key, 0)
                    + word_types[word.split(" ")[2]].get(key, 0)
                    for key in word_types[word.split(" ")[0]].keys()
                    | word_types[word.split(" ")[1]].keys()
                    | word_types[word.split(" ")[2]].keys()
                }
            )
        # If the word is not in word types, then add nan
        else:
            # TODO this might be too strict
            word_type_list.append(np.nan)

    word_type_column = {"word_type": word_type_list}
    word_type_df = pd.DataFrame(word_type_column)
    tf_idf_word_types = pd.concat([tf_idf, word_type_df], axis=1)

    # Fileter out rows with nan word_type
    tf_idf_word_types = tf_idf_word_types[
        tf_idf_word_types["word_type"].apply(lambda x: not pd.isna(x))
    ]

    verb_set = (
        set(["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]) if not include_verbs else set()
    )
    determiner_set = set(["DT", "PDT", "WDT"]) if not include_determiners else set()
    pronouns_set = set(["PRP", "PRP$", "WP", "WP$"]) if not include_pronouns else set()
    adverbs_set = set(["RB", "RBR", "RBS", "WRB"]) if not include_adverbs else set()
    numbers_set = set(["CD"]) if not include_numbers else set()

    exclude_set = verb_set | determiner_set | pronouns_set | adverbs_set | numbers_set

    if exclude_set:
        if verbose:
            print(
                "Excluding words with the following word types: {}".format(exclude_set)
            )
        # Exclude words if the frequency of the word type from exclude_set is
        # greater than 10% of the total word types of that word
        tf_idf_word_types = tf_idf_word_types[
            tf_idf_word_types["word_type"].apply(
                lambda x: not (
                    sum([x[word_type] for word_type in x if word_type in exclude_set])
                    / sum(x.values())
                    > 0.1
                )
            )
        ]
        # Exclude words if the word type is in exclude_set (This is too STRICT)
        # tf_idf_word_types = tf_idf_word_types[
        #     tf_idf_word_types["word_type"].apply(
        #         lambda x: not exclude_set & set(x.keys())
        #     )
        # ]

    new_column_order = ["word", "word_type", "tc", "tf", "dc", "idf", "tf_idf"]
    tf_idf_word_types = tf_idf_word_types[new_column_order]

    tf_idf = tf_idf.reset_index(drop=True)

    return tf_idf_word_types


def get_top_30_gender_number_dictionary():
    """Get a dictionary of the top 30 words (tf_idf) and their gender and number.

    Used in the relation extraction (spacy dependency tree parsing) algorithm
    when connecting pronouns to subjects (e.g. "he" to "Joseph"). For correctness,
    it's better to know the gender and number of the subject.

    Returns
    -------
    dict
    """
    # TODO: Currently manually created. Parsers do not know how to get Gender of
    # entities like Joseph and Jacob. Need to figure out how to automate this.
    top_30_words = [
        "joseph",
        "jacob",
        "abraham",
        "pharaoh",
        "esau",
        "duke",
        "abram",
        "master",
        "isaac",
        "sons",
        "laban",
        "years",
        "noah",
        "rachel",
        "earth",
        "father",
        "egypt",
        "daughters",
        "waters",
        "brethren",
        "brother",
        "son",
        "lot",
        "sarah",
        "abimelech",
        "dream",
        "daughter",
        "ark",
        "god",
        "king",
    ]
    top_30_words_genders = [
        "Masc",
        "Masc",
        "Masc",
        "Masc",
        "Masc",
        "Masc",
        "Masc",
        "Masc",
        "Masc",
        "Masc",
        "Masc",
        "Neut",
        "Masc",
        "Fem",
        "Neut",
        "Masc",
        "Neut",
        "Fem",
        "Neut",
        "Masc",
        "Masc",
        "Masc",
        "Neut",
        "Fem",
        "Masc",
        "Neut",
        "Fem",
        "Neut",
        "Masc",
        "Masc",
    ]
    # Add the Number (Sing or Plur) of the top_30_words
    top_30_words_numbers = [
        "Sing",
        "Sing",
        "Sing",
        "Sing",
        "Sing",
        "Sing",
        "Sing",
        "Sing",
        "Sing",
        "Plur",
        "Sing",
        "Plur",
        "Plur",
        "Plur",
        "Sing",
        "Sing",
        "Sing",
        "Plur",
        "Plur",
        "Plur",
        "Sing",
        "Sing",
        "Sing",
        "Sing",
        "Sing",
        "Sing",
        "Sing",
        "Sing",
        "Sing",
        "Sing",
    ]

    # Create a top_30_gender_number_dictionary
    top_30_gender_number_dictionary = dict(
        zip(top_30_words, zip(top_30_words_genders, top_30_words_numbers))
    )

    return top_30_gender_number_dictionary


def get_gospel_top_70_words_dictionary():
    """Get a dictionary of top 70 words of the Gospels with gender and number."""

    gospel_top_70_words_dictionary = {
        "unto": ("Neut", "Sing"),
        "shall": ("Neut", "Sing"),
        "jesus": ("Masc", "Sing"),
        "man": ("Masc", "Sing"),
        "son": ("Masc", "Sing"),
        "god": ("Masc", "Sing"),
        "things": ("Neut", "Plur"),
        "thy": ("Neut", "Sing"),
        "father": ("Masc", "Sing"),
        "lord": ("Masc", "Sing"),
        "disciples": ("Masc", "Plur"),
        "day": ("Masc", "Sing"),
        "men": ("Masc", "Plur"),
        "many": ("Masc", "Plur"),
        "house": ("Neut", "Sing"),
        "kingdom": ("Neut", "Sing"),
        "people": ("Masc", "Plur"),
        "world": ("Neut", "Sing"),
        "upon": ("Neut", "Sing"),
        "great": ("Masc", "Sing"),
        "john": ("Masc", "Sing"),
        "good": ("Neut", "Sing"),
        "peter": ("Masc", "Sing"),
        "may": ("Neut", "Sing"),
        "might": ("Neut", "Sing"),
        "among": ("Neut", "Sing"),
        "days": ("Masc", "Plur"),
        "way": ("Masc", "Sing"),
        "hand": ("Fem", "Sing"),
        "jews": ("Masc", "Plur"),
        "would": ("Neut", "Sing"),
        "life": ("Fem", "Sing"),
        "name": ("Neut", "Sing"),
        "pharisees": ("Masc", "Plur"),
        "mother": ("Fem", "Sing"),
        "time": ("Neut", "Sing"),
        "word": ("Neut", "Sing"),
        "children": ("Masc", "Plur"),
        "city": ("Fem", "Sing"),
        "jerusalem": ("Fem", "Sing"),
        "dead": ("Masc", "Plur"),
        "certain": ("Masc", "Sing"),
        "master": ("Masc", "Sing"),
        "chief": ("Masc", "Sing"),
        "spirit": ("Neut", "Sing"),
        "temple": ("Neut", "Sing"),
        "multitude": ("Fem", "Sing"),
        "hour": ("Fem", "Sing"),
        "priests": ("Masc", "Plur"),
        "simon": ("Masc", "Sing"),
        "bread": ("Neut", "Sing"),
        "galilee": ("Fem", "Sing"),
        "place": ("Neut", "Sing"),
        "whole": ("Neut", "Sing"),
        "christ": ("Masc", "Sing"),
        "servant": ("Masc", "Sing"),
        "scribes": ("Masc", "Plur"),
        "woman": ("Fem", "Sing"),
        "nothing": ("Neut", "Sing"),
        "brother": ("Masc", "Sing"),
        "earth": ("Neut", "Sing"),
        "thine": ("Neut", "Sing"),
        "king": ("Masc", "Sing"),
        "prophet": ("Masc", "Sing"),
        "hands": ("Fem", "Plur"),
        "pilate": ("Masc", "Sing"),
        "light": ("Neut", "Sing"),
        "sea": ("Fem", "Sing"),
        "mary": ("Fem", "Sing"),
        "wife": ("Fem", "Sing"),
    }

    return gospel_top_70_words_dictionary
