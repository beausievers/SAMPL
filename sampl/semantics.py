"""Semantic graphs."""
import random
import numpy as np
from scipy.spatial.distance import pdist, squareform


def connected_word_list(seed_word, keyed_vectors, n_iterations=5):
    """Get a list of closely related words."""
    all_words = [seed_word]
    for i in range(n_iterations):
        seed_word = random.sample(all_words, 1)
        nearest_neighbors = list(np.array(keyed_vectors.most_similar(seed_word))[:, 0])
        all_words += nearest_neighbors
    return all_words


def semantic_dsm(word_list, keyed_vectors):
    """Calculate a semantic dissimilarity matrix."""
    vectors = np.array([keyed_vectors.word_vec(word) for word in word_list])
    dsm = np.clip(pdist(vectors, metric="cosine"), 0, 1)
    return dsm


def semantic_dsm_safe(word_list, keyed_vectors):
    """Calculate a semantic dissimilarity matrix, safely."""
    vectors = []
    labels = []
    for word in word_list:
        try:
            vectors.append(keyed_vectors.word_vec(word))
        except:
            pass
        else:
            labels.append(word)
    vectors = np.array(vectors)
    matrix = pdist(vectors, metric="cosine")
    return (matrix, labels)


class SemanticGraph(object):
    """A graph where nodes are concepts and edges are association strengths.
    """

    def __init__(self, adj=None, labels=None, dsm=None, directed=True):
        """Create a SemanticGraph.

        Can initialize using either a weight adjacency matrix or a
        dissimilarity matrix (but not both).

        Arguments
        ---------
        adj : np.array (n x n) or list (length n) of lists (length n)
            Matrix of edge weights
        labels : list (length n)
            Node labels
        dsm : np.array (n x n)
            Matrix of node dissimilarities. Can also be 1-dimensional output,
            as from scipy.spatial.distance.pdist
        directed : bool
            Store whether the graph is directed so functions that operate on
            the graph can modify their behavior.
        """
        if dsm is None:
            self.adj = np.array(adj)
        else:
            if len(dsm.shape) == 1:
                dsm = squareform(dsm)
            adj = 1 - dsm
            np.fill_diagonal(adj, 0)
            self.adj = np.array(adj)
        self.labels = labels
        self.directed = directed
