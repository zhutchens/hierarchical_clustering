import random
import numpy as np
import pandas as pd
import nltk
import math
import re
import copy

from typing import Optional


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
            else:
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
    all_verse_words = []
    for verse in documents:
        verse_words = re.findall(r"\w+", verse)
        for i in range(len(verse_words)):
            verse_words[i] = verse_words[i].lower()
        all_verse_words.append(set(verse_words))

    for verse in documents:
        verse_words = re.findall(r"\w+", verse)
        for word in verse_words:
            word = word.lower()
            if word not in idf["word"].values and word not in stopwords_set:
                row = {}
                row["word"] = [word]
                row["idf"] = math.log(
                    N_documents
                    / len(
                        [True for verse_words in all_verse_words if word in verse_words]
                    )
                )
                row["dc"] = len(
                    [True for verse_words in all_verse_words if word in verse_words]
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

    tf = get_tf_for_documents(documents, sortby="word", skip_stopwords=skip_stopwords)
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

    # Create a word_type column in tf_idf, if the word is not in word_types, then it is nan
    word_type_column = {
        "word_type": [
            word_types[word] if word in word_types else np.nan
            for word in tf_idf["word"].values
        ]
    }
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
        print("Excluding words with the following word types: {}".format(exclude_set))
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
    top_30_words = ['joseph', 'jacob', 'abraham', 'pharaoh', 'esau', 'duke', 'abram',
        'master', 'isaac', 'sons', 'laban', 'years', 'noah', 'rachel',
        'earth', 'father', 'egypt', 'daughters', 'waters', 'brethren',
        'brother', 'son', 'lot', 'sarah', 'abimelech', 'dream', 'daughter',
        'ark', 'god', 'king']
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
    top_30_gender_number_dictionary = dict(zip(top_30_words, zip(top_30_words_genders, top_30_words_numbers)))

    return top_30_gender_number_dictionary