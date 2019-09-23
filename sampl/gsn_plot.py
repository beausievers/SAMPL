import copy
import itertools

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def plot_op_heatmap(op, vmin=0, vmax=1, annot=True, ax=None):
    sns.heatmap(
        op.graph.adj,
        xticklabels=op.graph.labels,
        yticklabels=op.graph.labels,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        cmap="viridis",
        ax=ax,
    )
    plt.xlabel("To")
    plt.ylabel("From")
    plt.title("Adjacency")


def sg_to_ng(sg):
    edge_bunch = []
    coords = list(
        itertools.combinations_with_replacement(list(range(sg.adj.shape[0])), 2)
    )
    for cx, cy in coords:
        if sg.adj[cx, cy] > 0:
            edge_bunch.append((sg.labels[cx], sg.labels[cy], sg.adj[cx, cy]))
    ng = nx.DiGraph()
    ng.add_nodes_from(sg.labels)
    ng.add_weighted_edges_from(edge_bunch)
    return ng


def plot_op_network(
    op,
    activations=None,
    node_size=6000,
    font_size=16,
    draw_node_labels=True,
    draw_edge_weights=True,
    custom_edges=None,
    draw_edge_colors=False,
    draw_node_activations=True,
    ax=None,
):
    """Plot the adjacency matrix of a GraphOperator using NetworkX.

    Args:
        op (GraphOperator): the semantic graph to plot
        activations (dict): keys are strings corresponding to node labels, and
            values are floats indicating the strength of activation.
        draw_node_labels (bool): if True, label nodes.
        draw_edge_weights (bool): if True, label edges with their weight.
        custom_edges (dict): keys are tuples referring to edges, e.g.,
            ('apple', 'banana'). Values are edge weights.
        draw_edge_colors (bool): if True, draw edge colors. Also makes +/-
            signs on edge labels easier to read.
        draw_node_activations (bool): if True, label nodes with activations.
        ax (matplotlib.axes.Axes): an Axes object to pass to NetworkX, or None.
    """
    from matplotlib.colors import LinearSegmentedColormap

    circular_layout_scale = 0.5
    weight_scaling_factor = 18

    sg = op.graph

    if custom_edges is None:
        ng = sg_to_ng(sg)
    else:
        ng = nx.DiGraph()
        ng.add_nodes_from(sg.labels)
        for key, value in custom_edges.items():
            ng.add_edge(key[0], key[1], weight=value)
    weights = [ng[u][v]["weight"] for u, v in ng.edges]
    widths = [weight_scaling_factor * w for w in weights]

    node_colors = [0] * len(ng.nodes)
    if activations is not None:
        for key, value in activations.items():
            node_colors[list(ng.nodes).index(key)] = value
    node_cmap = LinearSegmentedColormap.from_list(
        #'gray_red', [(.7, .7, .7), (.9, .5, .5)]
        "gray_red",
        [(0.85, 0.85, 0.85), (1.0, 0.6, 0.65)],
    )

    circular_pos = nx.circular_layout(ng, scale=circular_layout_scale)

    edge_colors = [0] * len(weights)
    if draw_edge_colors:
        edge_colors = weights

    nx.draw(
        ng,
        pos=circular_pos,
        width=widths,
        arrows=False,
        node_size=node_size,
        node_color=node_colors,
        cmap=node_cmap,
        vmin=0,
        vmax=1,
        edge_color=edge_colors,
        edge_cmap=plt.get_cmap("coolwarm"),
        edge_vmin=-1,
        edge_vmax=1,
        ax=ax,
        clip_on=False,
    )
    if draw_node_labels:
        pos_node_labels = copy.deepcopy(circular_pos)
        if draw_node_activations:
            for key in pos_node_labels.keys():
                if activations is not None and key in activations.keys():
                    pos_node_labels[key][1] += 0.03
        nx.draw_networkx_labels(ng, pos=pos_node_labels, font_size=font_size, ax=ax)

    if activations is not None and draw_node_activations:
        pos_activations = copy.deepcopy(circular_pos)
        for key in activations.keys():
            activations[key] = np.round(activations[key], 2)
        if draw_node_labels:
            for key in pos_activations.keys():
                pos_activations[key][1] -= 0.07
        nx.draw_networkx_labels(
            ng, pos=pos_activations, font_size=font_size, labels=activations, ax=ax
        )
    if draw_edge_weights:
        edge_labels = nx.get_edge_attributes(ng, "weight")
        for key in edge_labels.keys():
            edge_labels[key] = np.round(edge_labels[key], 2)
        if draw_edge_colors:
            for key, value in edge_labels.items():
                if value > 0:
                    edge_labels[key] = f"+ {value}"
                elif value < 0:
                    edge_labels[key] = f"- {np.abs(value)}"
                else:
                    edge_labels[key] = ""
        nx.draw_networkx_edge_labels(
            ng,
            pos=circular_pos,
            font_size=font_size,
            edge_labels=edge_labels,
            label_pos=0.5,
            ax=ax,
        )
