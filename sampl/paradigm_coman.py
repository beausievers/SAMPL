import copy
import itertools
import os
import time
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from scipy.sparse import csgraph
from sklearn import linear_model
from joblib import Parallel, delayed
from tqdm import tqdm

from sampl import gsn_api as gsn
from sampl import semantics as sem
from sampl import update as update
from sampl import agent as agent

clustered_adj = np.zeros(shape=(10, 10))
clustered_adj[0, (1, 2, 3)] = 1
clustered_adj[1, (0, 2, 3)] = 1
clustered_adj[2, (0, 1, 4)] = 1
clustered_adj[3, (0, 1, 4)] = 1
clustered_adj[4, (2, 3, 5)] = 1
clustered_adj[5, (4, 6, 7)] = 1
clustered_adj[6, (5, 8, 9)] = 1
clustered_adj[7, (5, 8, 9)] = 1
clustered_adj[8, (6, 7, 9)] = 1
clustered_adj[9, (6, 7, 8)] = 1

clustered = [
    [(0, 2), (1, 3), (4, 5), (7, 9), (6, 8)],
    [(1, 2), (3, 4), (5, 7), (6, 9)],
    [(0, 3), (2, 4), (5, 6), (7, 8)],
    [(0, 1), (8, 9)],
]

non_clustered_adj = np.zeros(shape=(10, 10))
non_clustered_adj[0, (8, 2, 3)] = 1
non_clustered_adj[1, (9, 2, 3)] = 1
non_clustered_adj[2, (0, 1, 4)] = 1
non_clustered_adj[3, (0, 1, 4)] = 1
non_clustered_adj[4, (2, 3, 5)] = 1
non_clustered_adj[5, (4, 6, 7)] = 1
non_clustered_adj[6, (5, 8, 9)] = 1
non_clustered_adj[7, (5, 8, 9)] = 1
non_clustered_adj[8, (6, 7, 0)] = 1
non_clustered_adj[9, (6, 7, 1)] = 1

non_clustered = [
    [(0, 2), (1, 3), (4, 5), (6, 8), (7, 9)],
    [(1, 2), (3, 4), (5, 7), (6, 9)],
    [(0, 3), (2, 4), (5, 6), (7, 8)],
    [(0, 8), (1, 9)],
]

clustered_graph = csgraph.csgraph_from_dense(clustered_adj)
non_clustered_graph = csgraph.csgraph_from_dense(non_clustered_adj)

clustered_shortest_paths = csgraph.shortest_path(clustered_graph)
non_clustered_shortest_paths = csgraph.shortest_path(non_clustered_graph)

# nc after, nc before, c after, c before
convergence_targets = np.array([0.66, 0.57, 0.606, 0.546])

alignment_targets = np.array([0.1, 0.078, 0.066, 0.043, 0.039, 0.107, 0.075, 0.077])


def empty_mind(n_categories=4, n_exemplars=4):
    n_rows = n_categories * n_exemplars + n_categories
    adj = np.zeros((n_rows, n_rows))
    labels = []
    for category_i in range(n_categories):
        for exemplar_i in range(n_exemplars):
            labels.append(f"cat{category_i}_ex{exemplar_i}")
    for category_i in range(n_categories):
        labels.append(f"cat{category_i}_label")
    return adj, labels


def study(agent, proportion, n_categories=4, n_exemplars=4):
    """Simulate study phase of Coman et al. (2016).

    Activate the specified proportion of category-exemplar pairs.

    Args:
        agent (Agent): the agent to study
        proportion (float): the proportion of items randomly selected for study
    """

    labels = agent.op.graph.labels
    assert len(labels) == n_categories * n_exemplars + n_categories
    cat_offset = n_categories * n_exemplars
    cat_ex_pairs = []
    for cat_i in range(n_categories):
        for ex_i in range(n_exemplars):
            pair = (cat_offset + cat_i, cat_i * n_exemplars + ex_i)
            cat_ex_pairs.append(pair)
    pair_is = np.random.choice(
        a=range(len(cat_ex_pairs)),
        size=int(len(cat_ex_pairs) * proportion),
        replace=False,
    )
    for pair_i in pair_is:
        act = np.zeros(len(labels))
        act[cat_ex_pairs[pair_i][0]] = 1.0
        act[cat_ex_pairs[pair_i][1]] = 1.0
        agent.op.activate_replace(act)


def converse(agent_a, agent_b, n_exchanges=25):
    agent_a.start_episode()
    agent_b.start_episode()

    for t in range(n_exchanges):
        words = agent_a.spontaneous_word_pair()
        agent_a.send(receivers=[agent_b], words=words)
        words = agent_b.spontaneous_word_pair()
        agent_b.send(receivers=[agent_a], words=words)

    agent_a.end_episode()
    agent_b.end_episode()


def pairwise_distances(agents, shortest_paths):
    distances = []
    binned_distances = {d: [] for d in np.unique(shortest_paths) if d > 0}
    pairs = list(itertools.combinations(range(len(agents)), 2))
    for i, j in pairs:
        agent_a = agents[i]
        agent_b = agents[j]
        d = dist.correlation(
            dist.squareform(agent_a.op.graph.adj), dist.squareform(agent_b.op.graph.adj)
        )
        net_d = shortest_paths[i, j]
        binned_distances[net_d].append(d)
    return (distances, binned_distances)


def network_sim(
    n_exchanges=100,
    learning_rate=0.5,
    dip_center=0.4,
    dip_width=0.4,
    y_min=-0.05,
    y_max=0.1,
    discount=0.3,
    words=None,
    keyed_vectors=None,
    study_proportion=0.5,
    print_log=False,
):
    """Run the Coman, Momennejad, Drach, & Geana (2016) paradigm.

    Args:
        n_exchanges:
    """
    chat_update = update.get_update(
        dip_center=dip_center, dip_width=dip_width, y_min=y_min, y_max=y_max
    )
    names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    agents_clustered = []
    agents_non_clustered = []
    for name in names:
        if words is not None and keyed_vectors is not None:
            dsm = sem.semantic_dsm(words, keyed_vectors)
            np.random.shuffle(dsm)
            g = sem.SemanticGraph(dsm=dsm, labels=words)
        else:
            adj, labels = empty_mind()
            g = sem.SemanticGraph(adj=adj, labels=labels)

        new_op = gsn.GraphOperator(
            graph=g,
            operate_fx=gsn.operate_recur,
            update_fx=chat_update,
            discount=discount,
        )
        new_agent = agent.Agent(
            name, op=new_op, learning_rate=learning_rate, print_log=print_log
        )
        if words is None and keyed_vectors is None:
            study(new_agent, study_proportion)

        agents_clustered.append(copy.deepcopy(new_agent))
        agents_non_clustered.append(copy.deepcopy(new_agent))

    agents_clustered_before = copy.deepcopy(agents_clustered)
    agents_non_clustered_before = copy.deepcopy(agents_non_clustered)

    for phase_i, phase in enumerate(clustered):
        if print_log:
            print(f"Phase {phase_i}")
        for i, j in phase:
            if print_log:
                print(
                    f"{agents_clustered[i].id_string} talks to "
                    f"{agents_clustered[j].id_string}"
                )
            converse(agents_clustered[i], agents_clustered[j], n_exchanges=n_exchanges)

    for phase_i, phase in enumerate(non_clustered):
        if print_log:
            print(f"Phase {i}")
        for i, j in phase:
            if print_log:
                print(
                    f"{agents_non_clustered[i].id_string} talks to "
                    f"{agents_non_clustered[j].id_string}"
                )
            converse(
                agents_non_clustered[i],
                agents_non_clustered[j],
                n_exchanges=n_exchanges,
            )

    return {
        "agents_clustered_before": agents_clustered_before,
        "agents_non_clustered_before": agents_non_clustered_before,
        "agents_clustered_after": agents_clustered,
        "agents_non_clustered_after": agents_non_clustered,
    }


def convergence(agents):
    """Average of mnemonic alignment (1 - distance) across a network."""
    pairs = list(itertools.combinations(range(len(agents)), 2))
    ds = []
    for i, j in pairs:
        agent_a = agents[i]
        agent_b = agents[j]
        ds.append(
            dist.correlation(
                dist.squareform(agent_a.op.graph.adj),
                dist.squareform(agent_b.op.graph.adj),
            )
        )
    ds = 1 - np.array(ds)
    return np.mean(ds)


def convergence_df(results, add_source=False):
    """Build a dataframe for plotting convergence results.

    Only makes sense if all of the results are generated using the same
    parameter settings.
    """
    d = {"Convergence": [], "Time": [], "Condition": []}
    for result in results:
        d["Convergence"].append(convergence(result["agents_clustered_before"]))
        d["Time"].append("Before")
        d["Condition"].append("Clustered")

        d["Convergence"].append(convergence(result["agents_clustered_after"]))
        d["Time"].append("After")
        d["Condition"].append("Clustered")

        d["Convergence"].append(convergence(result["agents_non_clustered_before"]))
        d["Time"].append("Before")
        d["Condition"].append("Non-clustered")

        d["Convergence"].append(convergence(result["agents_non_clustered_after"]))
        d["Time"].append("After")
        d["Condition"].append("Non-clustered")

    if add_source:
        d["Source"] = ["Simulation"] * len(d["Convergence"]) + ["CMDG 2016"] * 4
        d["Time"] += ["After", "Before", "After", "Before"]
        d["Condition"] += ["Non-clustered", "Non-clustered", "Clustered", "Clustered"]
        d["Convergence"] += list(convergence_targets)

    return pd.DataFrame(d)


def alignment(agent_a_before, agent_b_before, agent_a_after, agent_b_after):
    """Change in correlation distance between two agents."""
    d_before = dist.correlation(
        dist.squareform(agent_a_before.op.graph.adj),
        dist.squareform(agent_b_before.op.graph.adj),
    )
    d_after = dist.correlation(
        dist.squareform(agent_a_after.op.graph.adj),
        dist.squareform(agent_b_after.op.graph.adj),
    )
    return -1 * (d_after - d_before)


def alignment_hops(agents_before, agents_after, shortest_paths):
    alignments = []
    hops = []

    pairs = list(itertools.combinations(range(len(agents_before)), 2))
    for i, j in pairs:
        al = alignment(
            agents_before[i], agents_before[j], agents_after[i], agents_after[j]
        )
        n_hops = shortest_paths[i, j]
        alignments.append(al)
        hops.append(n_hops)
    return {"Alignment": alignments, "Hops": hops}


def alignment_df(results, add_source=False):
    output = {"Alignment": [], "Hops": [], "Condition": []}

    for result in results:
        clustered = alignment_hops(
            result["agents_clustered_before"],
            result["agents_clustered_after"],
            clustered_shortest_paths,
        )
        output["Alignment"] += clustered["Alignment"]
        output["Hops"] += clustered["Hops"]
        output["Condition"] += ["Clustered"] * len(clustered["Alignment"])
        non_clustered = alignment_hops(
            result["agents_clustered_before"],
            result["agents_clustered_after"],
            non_clustered_shortest_paths,
        )
        output["Alignment"] += non_clustered["Alignment"]
        output["Hops"] += non_clustered["Hops"]
        output["Condition"] += ["Non-clustered"] * len(non_clustered["Alignment"])

    if add_source:
        output["Source"] = ["Simulation"] * len(output["Alignment"]) + ["CMDG 2016"] * 8
        output["Hops"] += [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0]
        output["Condition"] += ["Clustered"] * 5 + ["Non-clustered"] * 3
        output["Alignment"] += list(alignment_targets)

    output["Hops"] = np.array(output["Hops"]).astype(int)
    return pd.DataFrame(output)


def cost(results):
    """Calculate the cost for results generated with a parameter set."""
    # This is all extremely unsophisticated munging, sorry future me.
    al_df = alignment_df(results)
    co_df = convergence_df(results)

    # Convergence targets
    co_nc_after_t = convergence_targets[0]
    co_nc_before_t = convergence_targets[1]
    co_c_after_t = convergence_targets[2]
    co_c_before_t = convergence_targets[3]

    # Targets as proportions of a standard
    # co_nc_after_tp = 1
    co_nc_before_tp = co_nc_before_t / co_nc_after_t
    co_c_after_tp = co_c_after_t / co_nc_after_t
    co_c_before_tp = co_c_before_t / co_nc_after_t

    # Actual values
    co_nc_after = co_df[
        (co_df["Condition"] == "Non-clustered") & (co_df["Time"] == "After")
    ]["Convergence"].mean()
    co_nc_before = co_df[
        (co_df["Condition"] == "Non-clustered") & (co_df["Time"] == "Before")
    ]["Convergence"].mean()
    co_c_after = co_df[
        (co_df["Condition"] == "Clustered") & (co_df["Time"] == "After")
    ]["Convergence"].mean()
    co_c_before = co_df[
        (co_df["Condition"] == "Clustered") & (co_df["Time"] == "Before")
    ]["Convergence"].mean()

    # Actual values as proportions of a standard
    # co_nc_after_p = 1
    co_nc_before_p = co_nc_before / co_nc_after
    co_c_after_p = co_c_after / co_nc_after
    co_c_before_p = co_c_before / co_nc_after

    # Alignment targets
    # As proportions
    alignment_targets_p = alignment_targets / alignment_targets[0]

    # Actual values
    al_c_1 = al_df[(al_df["Condition"] == "Clustered") & (al_df["Hops"] == 1)][
        "Alignment"
    ].mean()
    al_c_2 = al_df[(al_df["Condition"] == "Clustered") & (al_df["Hops"] == 2)][
        "Alignment"
    ].mean()
    al_c_3 = al_df[(al_df["Condition"] == "Clustered") & (al_df["Hops"] == 3)][
        "Alignment"
    ].mean()
    al_c_4 = al_df[(al_df["Condition"] == "Clustered") & (al_df["Hops"] == 4)][
        "Alignment"
    ].mean()
    al_c_5 = al_df[(al_df["Condition"] == "Clustered") & (al_df["Hops"] == 5)][
        "Alignment"
    ].mean()
    al_nc_1 = al_df[(al_df["Condition"] == "Non-clustered") & (al_df["Hops"] == 1)][
        "Alignment"
    ].mean()
    al_nc_2 = al_df[(al_df["Condition"] == "Non-clustered") & (al_df["Hops"] == 2)][
        "Alignment"
    ].mean()
    al_nc_3 = al_df[(al_df["Condition"] == "Non-clustered") & (al_df["Hops"] == 3)][
        "Alignment"
    ].mean()
    alignment_values = np.array(
        [al_c_1, al_c_2, al_c_3, al_c_4, al_c_5, al_nc_1, al_nc_2, al_nc_3]
    )
    # As proportions
    alignment_values_p = alignment_values / alignment_values[0]

    diffs_co = [
        co_nc_before_p - co_nc_before_tp,
        co_c_after_p - co_c_after_tp,
        co_c_before_p - co_c_before_tp,
    ]
    diffs_al = alignment_values_p - alignment_targets_p
    cost_co = np.mean(np.abs(diffs_co))
    cost_al = np.mean(np.abs(diffs_al))

    # Calculate the linear effect of Hops on Alignment in the Clustered
    # condition and compare to the actual result

    al_df_clustered = al_df[al_df["Condition"] == "Clustered"]
    if np.isnan(al_df_clustered["Alignment"].values).any():
        cost_reg = np.nan
    else:
        scale = (
            al_df_clustered[al_df_clustered["Hops"] == 1]["Alignment"].mean()
            / alignment_targets[0]
        )
        # Rise over run
        target_coef = (alignment_targets[4] * scale - alignment_targets[0] * scale) / 4

        reg = linear_model.LinearRegression()
        reg.fit(
            al_df_clustered["Hops"].values.reshape(-1, 1), al_df_clustered["Alignment"]
        )
        cost_reg = np.abs(reg.coef_[0] - target_coef)

    # Finally, take into account the absolute amount of alignment in both
    # the clustered and non-clustered conditions

    cost_al_abs = 1 - al_df_clustered[al_df_clustered["Hops"] == 1]["Alignment"].mean()

    c = np.average([cost_co, cost_al, cost_reg, cost_al_abs], weights=[1, 3, 2, 3])

    return c, cost_co, cost_al, cost_reg, cost_al_abs


def run_with_params(params, n_runs=10):
    y_min, y_max, dip_center, dip_width, discount, learning_rate = params
    results = []
    for _ in range(n_runs):
        results.append(
            network_sim(
                n_exchanges=50,
                learning_rate=learning_rate,
                dip_center=dip_center,
                dip_width=dip_width,
                y_min=y_min,
                y_max=y_max,
                discount=discount,
                words=None,
                keyed_vectors=None,
                study_proportion=0.7,
                print_log=False,
            )
        )
    return results


def parameter_search(
    grid, n_runs=5, n_jobs=4, start_i=0, end_i=None, memmap_path=None, overwrite=False
):
    start = time.time()
    print(
        f"Starting memmapped parameter search with "
        f"{n_runs} iterations per set at {start}..."
    )

    grid = np.array(grid)
    if end_i is None:
        end_i = grid.shape[0]

    print(f"grid.shape: {grid.shape}")
    print(f"start_i: {start_i}")
    print(f"end_i: {end_i}")

    if memmap_path is None:
        memmap_path = "coman.memmap"

    mode = "r+"
    if not os.path.exists(memmap_path):
        mode = "w+"
    output = np.memmap(
        memmap_path,
        dtype=np.single,
        shape=(grid.shape[0], grid.shape[1] + 5),
        mode=mode,
    )
    if overwrite:
        output[:, :] = 100

    def inner_helper(i, out):
        costs = cost(run_with_params(grid[i], n_runs=n_runs))
        out[i, 0 : grid[i].shape[0]] = grid[i]
        out[i, grid[i].shape[0] :] = costs

    Parallel(n_jobs=n_jobs)(
        delayed(inner_helper)(i, output) for i in tqdm(range(start_i, end_i))
    )

    end = time.time()
    print(
        f"Finished at {end}\n"
        f"# of param sets: {len(grid)}\t\t"
        f"Duration: {round((end - start) / 60, 2)} minutes"
    )
    return output
