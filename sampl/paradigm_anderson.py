"""Simulate Anderson, Bjork, & Bjork (1994)."""

import copy
import os
import random
import time

from sampl import gsn_api as gsn
from sampl import semantics as sem
from sampl import update

import scipy.spatial.distance as dist
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# In each category, the first 3 items are Set 1, and the last 3 are Set 2.
anderson_categories = {
    "fruit": ["orange", "nectarine", "pineapple", "banana", "cantaloupe", "lemon"],
    "leather": ["saddle", "gloves", "wallet", "shoes", "belt", "purse"],
    "tree": ["palm", "hickory", "willow", "poplar", "sequoia", "ash"],
    "profession": ["tailor", "florist", "farmer", "critic", "grocer", "clerk"],
    "drink": ["bourbon", "scotch", "tequila", "brandy", "gin", "rum"],
    "hobby": ["gardening", "coins", "stamps", "ceramics", "biking", "drawing"],
    "metal": ["chrome", "platinum", "magnesium", "mercury", "pewter", "tungsten"],
    "weapon": ["hammer", "fist", "lance", "rock", "arrow", "dagger"],
}
anderson_set_a = ["fruit", "leather", "tree", "profession"]
anderson_set_b = ["drink", "hobby", "metal", "weapon"]
anderson_strong = ["fruit", "leather", "drink", "hobby"]
anderson_weak = ["tree", "profession", "metal", "weapon"]


def anderson_words():
    output = []
    for category, words in anderson_categories.items():
        for word in words:
            output.append(word)
    for category in anderson_categories.keys():
        output.append(category)
    return output


def anderson_graph(keyed_vectors):
    """Parse anderson_categories to create a flat word list and SemanticGraph.

    Args:
        keyed_vectors (gensim.models.KeyedVectors): object containing vector
            representations of words
    """
    words = anderson_words()
    anderson_dsm = dist.squareform(
        sem.semantic_dsm(word_list=words, keyed_vectors=keyed_vectors)
    )
    anderson_graph = sem.SemanticGraph(dsm=anderson_dsm, labels=words)
    return words, anderson_graph


def study(op, category, exemplars, boost=0):
    """Simulate study by boosting category-to-exemplar edge weights."""
    category_i = op.graph.labels.index(category)

    for exemplar in exemplars:
        exemplar_i = op.graph.labels.index(exemplar)
        op.graph.adj[category_i, exemplar_i] = op.update_fx.x[3] + boost
        op.graph.adj[exemplar_i, category_i] = op.update_fx.x[3] + boost

    op.graph.adj = np.clip(op.graph.adj, a_min=0, a_max=1)


def cued_recall(op, category, exemplar):
    """Simulate cued recall by activating category and exemplar nodes."""
    category_i = op.graph.labels.index(category)
    exemplar_i = op.graph.labels.index(exemplar)
    activation = np.zeros(len(op.graph.labels))
    activation[category_i] = 1
    activation[exemplar_i] = 1
    op.activate_replace(activation)


def rif_weights(graph, rp_plus_pairs, rp_minus_pairs, nrp_pairs):
    """Collate edge weights given RP+, RP-, and NRP pair lists."""
    rp_plus_ix = np.array(
        [(graph.labels.index(p[0]), graph.labels.index(p[1])) for p in rp_plus_pairs]
    )
    rp_minus_ix = np.array(
        [(graph.labels.index(p[0]), graph.labels.index(p[1])) for p in rp_minus_pairs]
    )
    nrp_ix = np.array(
        [(graph.labels.index(p[0]), graph.labels.index(p[1])) for p in nrp_pairs]
    )
    rp_plus = graph.adj[rp_plus_ix[:, 0], rp_plus_ix[:, 1]]
    rp_minus = graph.adj[rp_minus_ix[:, 0], rp_minus_ix[:, 1]]
    nrp = graph.adj[nrp_ix[:, 0], nrp_ix[:, 1]]
    return [rp_plus, rp_minus, nrp]


def fit_cost_anderson(weights_t2):
    strong_rpp = 0.810
    strong_rpm = 0.403
    strong_nrp = 0.560
    weak_rpp = 0.662
    weak_rpm = 0.347
    weak_nrp = 0.410

    # The 'strong' words are more frequently occuring in the source corpus.
    # But we don't have a way to account for this in our model. So we just
    # take the mean of the strong and weak results as our target.
    target_rpp = np.mean([strong_rpp, weak_rpp])
    target_rpm = np.mean([strong_rpm, weak_rpm])
    target_nrp = np.mean([strong_nrp, weak_nrp])

    err_rpp = np.median(np.absolute(weights_t2[0] - target_rpp))
    err_rpm = np.median(np.absolute(weights_t2[1] - target_rpm))
    err_nrp = np.median(np.absolute(weights_t2[2] - target_nrp))

    cost = np.sum([err_rpp, err_rpm, err_nrp])
    return cost


def anderson_practice_lists():
    """Parse anderson_categories into RPP, RPM, and NRP lists.

    There is a complex counter-balancing scheme for retrieval practice. Ss were
    split into 4 groups: A1, A2, B1, B2. Group A1 practiced Set A Set 1, Group
    A2 practiced Set A Set 2, and so on.
    """
    practice_lists = []
    for set_rp in [anderson_set_a, anderson_set_b]:
        for word_ix in [[0, 1, 2], [3, 4, 5]]:
            rpp = []
            rpm = []
            nrp = []
            for cat in anderson_categories.keys():
                if cat in set_rp:
                    for word_i, word in enumerate(anderson_categories[cat]):
                        if word_i in word_ix:
                            rpp.append((cat, word))
                        else:
                            rpm.append((cat, word))
                else:
                    for word in anderson_categories[cat]:
                        nrp.append((cat, word))
            practice_lists.append({"rpp": rpp, "rpm": rpm, "nrp": nrp})
    return practice_lists


def run_anderson_counterbalanced(
    op,
    practice_lists=anderson_practice_lists(),
    categories=anderson_categories,
    n_recalls=3,
):
    results = []
    for practice_list in practice_lists:
        results.append(
            run_anderson_single(copy.deepcopy(op), practice_list, categories, n_recalls)
        )
    return results


def run_anderson_single(op, practice_list, categories=anderson_categories, n_recalls=3):
    """Simulate Anderson, Bjork, & Bjork (1994) experiment 1."""
    for category, exemplars in categories.items():
        study(op, category, exemplars)
    op_studied = copy.deepcopy(op)

    for _ in range(n_recalls):
        rpp = practice_list["rpp"]
        practice_order = np.random.permutation(len(rpp))
        for i in practice_order:
            try:
                cued_recall(op, rpp[i][0], rpp[i][1])
            except:
                print("Error during cued recall.")
                raise

    return {"op": op, "op_studied": op_studied, "practice_list": practice_list}
