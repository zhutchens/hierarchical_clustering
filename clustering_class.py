import random
import numpy as np
import copy

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

random.seed(1)


class HierarchicalClustering:
    """
    Clustering class for hierarchical word embedding clustering.
    """

    def __init__(
        self,
        word_embedding,
        list_of_vectors,
        chosen_indices: set,
        initial_proximity: float = 3.0,
        proximity_reduc: float = 0.5,
        initial_proximity_inc: float = 2.0,
        verbose: bool = False,
    ):
        # set word embedding & vectors
        self.word_embedding = word_embedding
        self.list_of_vectors = list_of_vectors
        self.chosen_indices = chosen_indices
        self.n_vectors = len(list_of_vectors)

        # set proximity constants
        self.initial_proximity = initial_proximity
        self.proximity_reduc = proximity_reduc
        self.initial_proximity_inc = initial_proximity_inc

        self.verbose = verbose

        # set the distance matrix with respect to the
        # given `list_of_vectors`
        self.set_distance_matrix_and_vectors()

    def set_distance_matrix_and_vectors(self, list_of_vectors=None, chosen_indices=None):
        """
        Sets the `distance_matrix` of all vectors to each other.
        If a new `list_of_vectors` is provided, then it is updated and
        the `distance_matrix` is set with respect to it.
        """

        if list_of_vectors is not None:
            if chosen_indices is None:
                raise ValueError(f"When setting the list_of_vectors, please provide the \
                    chosen_indices in the word_embedding, even if it's all.")
            self.list_of_vectors = list_of_vectors
            self.chosen_indices = chosen_indices

        self.distance_matrix = np.zeros((self.n_vectors, self.n_vectors))

        for index in range(self.n_vectors):
            self.distance_matrix[index] = np.sqrt(
                np.sum(
                    np.square(self.list_of_vectors - self.list_of_vectors[index]),
                    axis=1,
                )
            )
            if self.verbose and (index + 1) % 1000 == 0:
                print(
                    "Setting distance matrix progress: ",
                    (index + 1) / self.n_vectors * 100,
                    "%",
                )
        if self.verbose:
            print("Finished setting distance matrix!")

    def which_in_proximity(self, index: int, current_proximity):
        """
        Returns a numpy array of indicies of all vectors which are in
        proximity of the vector with index `index` with respect to the `current_proximity`.
        """

        return np.where(self.distance_matrix[index] < current_proximity)[0]

    def proximity_spread(
        self,
        new_main_head: int,
        used_indices: set,
        not_used_indices: set,
        current_proximity: float,
    ):
        """
        Executes a proximity spread around the given `new_main_head` using the
        `current_proximity`, `used_indices` and `not_used_indices`.
        Returns a set of the found proximity elements and updated `used_indices` and `not_used_indices`.
        """
        used_indices.add(new_main_head)

        main_proximity_elements = set()

        head_elements = set()
        head_elements.add(new_main_head)
        while True:

            new_proximity_elements = set()

            for element in head_elements:
                new_proximity_elements = new_proximity_elements.union(
                    set(self.which_in_proximity(element, current_proximity))
                )
            # print("New proximity elements: ", new_proximity_elements)
            # remove already used_indices elements
            new_proximity_elements = new_proximity_elements.difference(used_indices)
            # print("Got new proximity elements: \n", new_proximity_elements)

            if len(new_proximity_elements) == 0:
                break

            # add to proximity elements
            main_proximity_elements = main_proximity_elements.union(
                new_proximity_elements
            )

            head_elements = new_proximity_elements
            used_indices = used_indices.union(head_elements)
            not_used_indices = not_used_indices.difference(head_elements)
            current_proximity = current_proximity * self.proximity_reduc

        return main_proximity_elements, used_indices, not_used_indices

    def get_sorted_not_used_by_relevance(
        self,
        used: set,
        not_used: set,
        level_initial_proximity: float,
    ):
        """
        Returns a sorted dictionary of `not_used_relevance`. The sorting is
        executed with respect to the amount of vectors in the proximity spread.
        """
        used_indices = copy.copy(used)
        not_used_indices = copy.copy(not_used)

        not_used_relevance = {}

        # if self.verbose:
        #     print(f"--- Starting to sort not used of length {len(not_used_indices)}.")

        for element in not_used_indices:
            element_proximity_spread, _, _ = self.proximity_spread(
                new_main_head=element,
                used_indices=used_indices,
                not_used_indices=not_used_indices,
                current_proximity=level_initial_proximity,
            )

            not_used_relevance[element] = element_proximity_spread

        #if self.verbose:
            #print(f"--- Finished sorting not used!")

        return {
            k: v
            for k, v in sorted(
                not_used_relevance.items(), key=lambda item: len(item[1]), reverse=True
            )
        }

    def get_better_list_of_hierarchical_orders(self):
        """
        Returns the better hierarchical clustering, i.e. a list of dictionaries.
        Each index of the list is a dictionary representing an order of
        the hierarchical clustering, starting from the lowest to highest.
        The
        """

        list_of_hierarchical_levels = []
        list_of_hierarchical_levels_w = []

        not_used = set(self.chosen_indices)
        used = set(range(self.n_vectors)).difference(not_used)
        print(f"Starting with the {len(used)} used and {len(not_used)} not used indices.")

        level_initial_proximity = self.initial_proximity

        if self.verbose:
            hierarchical_level_index = 0

        # iterate over hierarchical levels
        while True:
            if self.verbose:
                print(f"Starting {hierarchical_level_index+1}. hierarchical level.")
                hierarchical_level_index += 1

            # dictionary with key main heads and values the proximity elements
            hierarchical_level = {}
            hierarchical_level_w = {}


            # iterate over heads
            while True:

                sorted_not_used_relevance = self.get_sorted_not_used_by_relevance(
                    used=used,
                    not_used=not_used,
                    level_initial_proximity=level_initial_proximity,
                )

                sorted_not_used_list = list(sorted_not_used_relevance.keys())

                new_main_head = sorted_not_used_list.pop(0)
                not_used.remove(new_main_head)
                used.add(new_main_head)

                main_proximity_elements = sorted_not_used_relevance[
                    new_main_head
                ].difference(used)
                
                if self.verbose:
                    print(f"--- Finished sorting: len(not_used)={len(not_used)}, new_proximity_size = {len(main_proximity_elements)}")

                used = used.union(main_proximity_elements)
                not_used = not_used.difference(main_proximity_elements)

                for proximity_element in main_proximity_elements:
                    sorted_not_used_list.remove(proximity_element)

                hierarchical_level[new_main_head] = main_proximity_elements
                hierarchical_level_w[
                    self.word_embedding.index_to_key[new_main_head]
                ] = set(
                    [
                        self.word_embedding.index_to_key[el]
                        for el in main_proximity_elements
                    ]
                )

                if len(not_used) == 0:
                    list_of_hierarchical_levels.append(hierarchical_level)
                    list_of_hierarchical_levels_w.append(hierarchical_level_w)
                    break

            last_level_heads = set(list_of_hierarchical_levels[-1].keys())

            self.visualize_hierarchical_level(reduced_vectors, last_level_heads, list_of_hierarchical_levels[-1].values())
            # break if there was only one last level head
            if self.verbose and len(last_level_heads) == 1:
                print("Hierarchical clustering finished!")
                break
            # increase level_initial_proximity
            level_initial_proximity = (
                level_initial_proximity + self.initial_proximity_inc
            )
            # change used to all except last level head
            used = set(range(self.n_vectors)).difference(last_level_heads)
            # change not_used to the last level head
            not_used = last_level_heads

        return list_of_hierarchical_levels, list_of_hierarchical_levels_w

    def get_list_of_hierarchical_orders(self):
        """
        Returns the hierarchical clustering, i.e. a list of dictionaries.
        Each index of the list is a dictionary representing an order of
        the hierarchical clustering, starting from the lowest to highest.
        The
        """

        raise ValueError("This function has been deprecated. It is worse and outdated. \
            Please use the get_better_list_of_hierarchical_orders method instead.")

        list_of_hierarchical_levels = []
        list_of_hierarchical_levels_w = []

        used = set()
        not_used = set(range(self.n_vectors))

        level_initial_proximity = self.initial_proximity

        if self.verbose:
            hierarchical_level_index = 0

        # iterate over hierarchical levels
        while True:
            if self.verbose:
                print(f"Starting {hierarchical_level_index+1}. hierarchical level ")
                hierarchical_level_index += 1

            # dictionary with key main heads and values the proximity elements
            hierarchical_level = {}
            hierarchical_level_w = {}

            # iterate over heads
            while True:
                proximity = level_initial_proximity

                # new_main_head = random.sample(not_used, 1)[0]
                # not_used.remove(new_main_head)
                new_main_head = not_used.pop()

                main_proximity_elements, used, not_used = self.proximity_spread(
                    new_main_head=new_main_head,
                    used_indices=used,
                    not_used_indices=not_used,
                    current_proximity=proximity,
                )

                hierarchical_level[new_main_head] = main_proximity_elements
                hierarchical_level_w[
                    self.word_embedding.index_to_key[new_main_head]
                ] = set(
                    [
                        self.word_embedding.index_to_key[el]
                        for el in main_proximity_elements
                    ]
                )

                if len(not_used) == 0:
                    list_of_hierarchical_levels.append(hierarchical_level)
                    list_of_hierarchical_levels_w.append(hierarchical_level_w)
                    break

            last_level_heads = set(list_of_hierarchical_levels[-1].keys())

            # break if there was only one last level head
            if self.verbose and len(last_level_heads) == 1:
                print("Hierarchical clustering finished!")
                break
            # increase level_initial_proximity
            level_initial_proximity = (
                level_initial_proximity + self.initial_proximity_inc
            )
            # change used to all except last level head
            used = set(range(self.n_vectors)).difference(last_level_heads)
            # change not_used to the last level head
            not_used = last_level_heads

        return list_of_hierarchical_levels, list_of_hierarchical_levels_w

    def visualize_hierarchical_level(self, vectors_to_visualize, heads, children):
        pca_alg = PCA(n_components=2)

        pc = pca_alg.fit_transform(vectors_to_visualize)

        head_pc = [pc[i] for i in heads]
        children_pc = [pc[i] for i in children]

        #ax = fig.add_subplot(projection = '3d')
        plt.scatter(head_pc[:,0], head_pc[:,1], marker='s', color='r')
        plt.scatter(children_pc[:,0], children_pc[:,1])
        plt.show()        

import gensim
import gensim.downloader
import spacy
import nltk

glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')


from nltk.corpus import wordnet as wn
wordnet_chosen_words=[]
for word in glove_vectors.index_to_key[90:5000]:
    tmp = wn.synsets(word)
    if len([t.pos() for t in tmp if t.pos()=='n'])>=1:
        wordnet_chosen_words.append(word)

wordnet_chosen_indices = set([glove_vectors.key_to_index[word] for word in wordnet_chosen_words])

vocab_size = 5000

proximity_const = 1
reducing_coef_const = 0.7
increasing_proximity_const = 0.2

# normed vectors
reduced_vectors = glove_vectors.get_normed_vectors()[:vocab_size]

hierarchical_clustering = HierarchicalClustering(
    word_embedding=glove_vectors,
    list_of_vectors=reduced_vectors,
    chosen_indices=wordnet_chosen_indices,
    initial_proximity=proximity_const,
    proximity_reduc=reducing_coef_const,
    initial_proximity_inc=increasing_proximity_const,
    verbose=True
)


better_hier_list, better_hier_list_w = hierarchical_clustering.get_better_list_of_hierarchical_orders()

# chosen_words = np.asarray([reduced_vectors[i] for i in wordnet_chosen_indices])
# hierarchical_clustering.visualize_vectors(chosen_words)