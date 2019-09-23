import copy
import os
import random
import time

from sampl import gsn_api as gsn
from sampl import semantics as sem
from sampl import update

import numpy as np
from scipy.stats import bayes_mvs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from joblib import Parallel, delayed, dump, load
from tqdm import tqdm


class Stimulus:
    def __init__(self, label=None, stim_type=None):
        self.label = label
        self.stim_type = stim_type


class Triplet_rcrt:
    def __init__(self):
        self.first = self.second = [
            Stimulus(stim_type="context_rcrt"),
            Stimulus(stim_type="context_rcrt"),
            Stimulus(stim_type="target_rcrt"),
        ]


class Triplet_ncrt:
    def __init__(self):
        self.first = [
            Stimulus(stim_type="context_1_ncrt"),
            Stimulus(stim_type="context_1_ncrt"),
            Stimulus(stim_type="target_ncrt"),
        ]
        self.second = [
            Stimulus(stim_type="context_1_ncrt"),
            Stimulus(stim_type="context_1_ncrt"),
            self.first[2],
        ]


class Triplet_rcnt:
    def __init__(self):
        self.first = [
            Stimulus(stim_type="context_rcnt"),
            Stimulus(stim_type="context_rcnt"),
            Stimulus(stim_type="target_1_rcnt"),
        ]
        self.second = [
            self.first[0],
            self.first[1],
            Stimulus(stim_type="target_2_rcnt"),
        ]


class Triplet_ncnt:
    def __init__(self):
        self.first = [
            Stimulus(stim_type="context_1_ncnt"),
            Stimulus(stim_type="context_1_ncnt"),
            Stimulus(stim_type="target_1_ncnt"),
        ]
        self.second = [
            Stimulus(stim_type="context_2_ncnt"),
            Stimulus(stim_type="context_2_ncnt"),
            Stimulus(stim_type="target_2_ncnt"),
        ]


def run_smith(
    n_klass_reps=8,
    dip_center=0.1,
    dip_width=0.1,
    y_min=-0.025,
    y_max=0.65,
    discount=0.2,
    window_size=3,
    plot_steps=False,
    print_triplets=False,
    debug=False,
):
    if debug:
        print("run_smith() params:")
        print(locals())
        print("building study list...")
    triplets = []
    for klass in [Triplet_rcrt, Triplet_ncrt, Triplet_rcnt, Triplet_ncnt]:
        for _ in range(n_klass_reps):
            triplets.append(klass())

    study_list = []
    for triplet in random.sample(triplets, len(triplets)):
        study_list += triplet.first
    for triplet in random.sample(triplets, len(triplets)):
        study_list += triplet.second

    for i, s in enumerate(set(study_list)):
        s.label = i

    n_labels = len([s.label for s in set(study_list)])

    if debug:
        print("building operator...")
    smith_graph = sem.SemanticGraph(
        np.zeros((n_labels, n_labels)), [s.label for s in set(study_list)]
    )
    smith_update = update.get_update(
        dip_center=dip_center, dip_width=dip_width, y_min=y_min, y_max=y_max
    )
    smith_op = gsn.GraphOperator(
        graph=copy.deepcopy(smith_graph),
        operate_fx=gsn.operate_recur,
        update_fx=smith_update,
        discount=discount,
    )

    if debug:
        print("iterating over study list...")
    previous_heatmap = smith_op.graph.adj
    for i in range(len(study_list) - (window_size - 1)):
        if debug:
            print(f"iteration {i} of {len(study_list) - (window_size - 1)}")
        window = study_list[i : i + window_size]
        activation = np.zeros(len(smith_op.graph.labels))
        for s in window:
            activation[s.label] = 1
        smith_op.activate_replace(activation)
        if plot_steps:
            sns.heatmap(smith_op.graph.adj, cmap="viridis")
            plt.show()
            sns.heatmap(
                smith_op.graph.adj - previous_heatmap, cmap="coolwarm", center=0
            )
            previous_heatmap = smith_op.graph.adj
            plt.show()
        if print_triplets:
            print([s.label for s in window])

    return smith_op, study_list


def analyze_smith(smith_op, study_list):
    stim_types = list(set([s.stim_type for s in study_list]))
    stim_type_labels = {}
    for stim_type in stim_types:
        stim_type_labels[stim_type] = set()
        for s in study_list:
            if stim_type == s.stim_type:
                stim_type_labels[stim_type].add(s.label)
    in_degrees = [
        smith_op.graph.adj[:, i].sum() for i in range(smith_op.graph.adj.shape[0])
    ]
    stim_type_in_degrees = {}
    for key, value in stim_type_labels.items():
        stim_type_in_degrees[key] = [in_degrees[i] for i in value]

    twice_presented_targets = (
        stim_type_in_degrees["target_ncrt"] + stim_type_in_degrees["target_rcrt"]
    )
    once_presented_targets = (
        stim_type_in_degrees["target_1_ncnt"]
        +
        # stim_type_in_degrees['target_2_ncnt'] +
        stim_type_in_degrees["target_1_rcnt"]
        # stim_type_in_degrees['target_2_rcnt']
    )

    # Item repetition effect
    ire = np.array(twice_presented_targets) - np.array(once_presented_targets)

    # Context repetition effect, first target
    cre_t1 = np.array(stim_type_in_degrees["target_1_rcnt"]) - np.array(
        stim_type_in_degrees["target_1_ncnt"]
    )

    # Context repetition effect, second target
    cre_t2 = np.array(stim_type_in_degrees["target_2_rcnt"]) - np.array(
        stim_type_in_degrees["target_2_ncnt"]
    )

    df = {
        "effect": (
            ["Item repetition effect"] * len(ire)
            + ["Context repetition, 1st target"] * len(cre_t1)
            + ["Context repetition, 2nd target"] * len(cre_t2)
        ),
        "contrast": list(ire) + list(cre_t1) + list(cre_t2),
    }
    return pd.DataFrame(df)


def smith_cost(df, debug=False):
    ire = df[df["effect"] == "Item repetition effect"]["contrast"]
    cre1 = df[df["effect"] == "Context repetition, 1st target"]["contrast"]
    cre2 = df[df["effect"] == "Context repetition, 2nd target"]["contrast"]

    ire_mvs = bayes_mvs(ire)
    cre1_mvs = bayes_mvs(cre1)
    cre2_mvs = bayes_mvs(cre2)

    if debug:
        print(f"ire_mvs: {ire_mvs}")
        print(f"cre1_mvs: {cre1_mvs}")
        print(f"cre2_mvs: {cre2_mvs}")

    ratio_top = cre1_mvs[0][0]
    ratio_bottom = ire_mvs[0][0]
    if np.isnan(ratio_top):
        ratio_top = 0
    if np.isnan(ratio_bottom):
        ratio_bottom = 0.00001

    ratio = ratio_top / ratio_bottom
    ratio_cost = abs(ratio - 0.744)  # cost = 0 if ratio is an exact match

    # add 1 if ire and cre1 CIs don't overlap
    # ire_cre1_overlap_cost = 1 - (
    #    (cre1_mvs[0][1][0] < ire_mvs[0][1][0]) and
    #    (cre1_mvs[0][1][1] < ire_mvs[0][1][1]) and
    #    (cre1_mvs[0][1][1] > ire_mvs[0][1][0])
    # )

    # add 1 if zero is in cre1 CI
    cre1_ci_lo = cre1_mvs[0][1][0]
    cre1_ci_hi = cre1_mvs[0][1][1]
    if np.isnan(cre1_ci_lo) or np.isnan(cre1_ci_hi):
        cre1_zero_cost = 1
    else:
        cre1_zero_cost = int(cre1_ci_lo < 0 < cre1_ci_hi)

    # add 1 if zero is not in cre2 CI
    cre2_ci_lo = cre2_mvs[0][1][0]
    cre2_ci_hi = cre2_mvs[0][1][1]
    if np.isnan(cre2_ci_lo) or np.isnan(cre2_ci_hi):
        if np.isclose(cre2.mean(), 0):
            cre2_zero_cost = 0
        else:
            cre2_zero_cost = 1
    else:
        cre2_zero_cost = 1 - int(cre2_ci_lo <= 0 <= cre2_ci_hi)

    cost = ratio_cost + cre1_zero_cost + cre2_zero_cost

    if debug:
        print(f"Ratio cost: {ratio_cost}")
        print(f"CRE1 zero cost: {cre1_zero_cost}")
        print(f"CRE2 zero cost: {cre2_zero_cost}")
        print(f"Total cost: {cost}")

    return cost


def run_with_params(params):
    y_min, y_max, dip_center, dip_width, discount = params
    smith_op, study_list = run_smith(
        n_klass_reps=12,
        dip_center=dip_center,
        dip_width=dip_width,
        y_min=y_min,
        y_max=y_max,
        discount=discount,
        window_size=3,
        debug=False,
    )
    return smith_op, study_list


def parameter_search(grid, n_sims=30, n_jobs=4, bar=False):
    start = time.time()
    print(
        f"Started parameter search over {len(grid)} sets with"
        f" {n_sims} iterations per set at {start}..."
    )

    def inner_helper(params):
        results = [run_with_params(params) for _ in range(n_sims)]
        dfs = [analyze_smith(r[0], r[1]) for r in results]
        return np.mean([smith_cost(df) for df in dfs])

    if bar:
        costs = Parallel(n_jobs=n_jobs)(
            delayed(inner_helper)(params) for params in tqdm(grid)
        )
    else:
        costs = Parallel(n_jobs=n_jobs)(
            delayed(inner_helper)(params) for params in grid
        )

    end = time.time()
    print(
        f"Finished at {end}\n"
        f"# of param sets: {len(grid)}\t\t"
        f"Duration: {round((end - start) / 60, 2)} minutes"
    )

    return costs


def parameter_search_mem(
    grid, n_sims=30, n_jobs=4, start_i=0, end_i=None, memmap_path=None, overwrite=False
):
    start = time.time()
    print(
        f"Memmapped parameter search with {n_sims} iterations per set started"
        f" at {start}..."
    )

    grid = np.array(grid)
    if end_i is None:
        end_i = grid.shape[0]

    print(f"grid.shape: {grid.shape}")
    print(f"start_i: {start_i}")
    print(f"end_i: {end_i}")

    if memmap_path is None:
        memmap_path = "smith.memmap"

    mode = "r+"
    if not os.path.exists(memmap_path):
        mode = "w+"
    output = np.memmap(
        memmap_path,
        dtype=np.single,
        shape=(grid.shape[0], grid.shape[1] + 1),
        mode=mode,
    )
    if overwrite:
        output[:, :] = 100

    def inner_helper(i, out):
        results = [run_with_params(grid[i]) for _ in range(n_sims)]
        dfs = [analyze_smith(r[0], r[1]) for r in results]
        out[i, 0 : grid.shape[1]] = grid[i]
        out[i, grid.shape[1]] = np.mean([smith_cost(df) for df in dfs])

    Parallel(n_jobs=n_jobs)(
        delayed(inner_helper)(i, output) for i in tqdm(range(start_i, end_i))
    )

    end = time.time()
    print(
        f"Finished at {end}\n"
        f"# of param sets: {len(grid)}\t\t"
        f"Duration: {round((end - start) / 60, 2)} minutes\n"
    )

    return output
