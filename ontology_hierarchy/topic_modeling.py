import contractions
import nltk
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    """Normalize the given document.

    Normalizes the given document by lower casing, removing special characters,
    removing stopwords, and stemming.

    Parameters:
    ----------
    doc: str
        The document to normalize.
    
    Returns:
    -------
    doc: str
        The normalized document.
    """
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    doc = contractions.fix(doc)
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    #filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

def kmeans_tfidf_clustering(chapters, num_topics):
    """Use KMeans clustering to cluster the given chapters into topics.
    
    Parameters:
    ----------
    chapters: list
        A list of chapters to cluster.
    num_topics: int
        The number of topics to cluster the chapters into.
    
    Returns:
    -------
    clusters: dict
        A dictionary of clusters and their chapters.
    key_terms_per_cluster: dict
        A dictionary of clusters and their key terms.
    """
    # Normalize the corpus.
    normalize_corpus = np.vectorize(normalize_document)
    norm_tr_corpus = normalize_corpus(list(chapters))

    # Extract tf-idf features.
    tfidf_vectorizer = TfidfVectorizer(
    max_features=200000,
    use_idf=True,
    ngram_range=(1, 2), 
    min_df=0.001, 
    max_df=0.99, 
    stop_words=stop_words)

    tfidf_matrix = tfidf_vectorizer.fit_transform(norm_tr_corpus)

    # Initialize KMeans clustering.
    km_tfidf = KMeans(
        n_clusters=num_topics, 
        max_iter=10000, 
        n_init=100, 
        verbose=0,
        random_state=42)
    
    # Fit the tf-idf features.
    km_tfidf.fit(tfidf_matrix)
    # Get the clusters.
    km_tfidf_clusters = km_tfidf.labels_.tolist()

    # Get the cluster centers.
    ordered_centroids = km_tfidf.cluster_centers_.argsort()[:, ::-1]
    feature_names = tfidf_vectorizer.get_feature_names_out()

    clusters = {}
    key_terms_per_cluster = {}

    # Extract the key terms per cluster and the chapters per cluster.
    for cluster in range(num_topics):
        key_features = [feature_names[index] for index in ordered_centroids[cluster, :50]]
        print('CLUSTER #'+str(cluster+1))
        print('Key Features:', key_features)

        cluster_chapter_indices = [i for i in range(len(km_tfidf_clusters)) if km_tfidf_clusters[i] == cluster]
        print('Cluster Chapters:', cluster_chapter_indices)

        clusters[cluster+1] = cluster_chapter_indices
        key_terms_per_cluster[cluster+1] = key_features
    
    return clusters, key_terms_per_cluster
    