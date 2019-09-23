import copy
import numpy as np


class GraphOperator(object):
    """Wrapper object for functions that activate and reweight a semantic graph
    """

    def __init__(self, graph, operate_fx, update_fx, discount):
        """Create a wrapper object for graph activation and reweighting.

        Arguments
        ---------
        graph : SemanticGraph
        operate_fx : function [graph, activations, update_fx, discount] ->
                [SemanticGraph]
            Function describing how to activate and reweight the graph
        update_fx : function [0, 1] -> float
            Scaling function for edge reweighting
        discount : float
            Discount coefficient (gamma) for spreading activation in operate_fx
        """
        self.graph = graph
        self.operate_fx = operate_fx
        self.update_fx = update_fx
        self.discount = discount

    def activate(self, activations):
        """Call operate_fx with stored arguments and return result."""
        return self.operate_fx(
            self.graph.adj, activations, self.update_fx, self.discount
        )

    def activate_replace(self, activations):
        """Replace self.graph with operate_fx result."""
        self.graph.adj = self.activate(activations)
        return self.graph.adj


def operate_recur(adj, activations, update, discount):
    """Wrapper for spreading activation and reweighting."""
    new_activations = spread_recur(adj, activations, update, discount)
    return update_recur(adj, new_activations, update)


def spread_recur(adj, activations, update, discount=0.8, new_adj=None, debug=False):
    """Recursively calculate downstream activations.

    Downstream node activation = ReLU(node activation * edge weight * discount),
    with two caveats: (1) Each node can only be activated once; i.e., loops are
    disallowed. (2) Activations below the lowest update threshold are zeroed out,
    so large numbers of small activations don't create ever-larger activations
    in distant downstream nodes; i.e., prevent blow-up.

    (1) can be thought of as requiring that all downstream activations take
    place within the span of each neuron's refractory period, and (2) casts the
    lowest x value of the update function as the activation threshold below which
    neurons don't fire.

    Args:
        adj (np.array, 2D): Weighted adjacency matrix.
        activations (np.array, 1D): Vector of node-wise activation strengths.
        update (function from sp.interp1d): update function for reweighting edges.
        discount (float): discount coefficient.
        new_adj (np.array, 2D): Pruned adjacency matrix used in recursion.
            In most cases, when this function is called new_adj should be None.
        debug (bool): Print debug information.
    """
    if new_adj is None:
        new_adj = copy.copy(adj)
    else:
        new_adj = copy.copy(new_adj)
    if debug:
        print(f"Debugging info for spread_recur()")
        print(f"activations, before zeroing: {activations}")

    # If activations are below threshold, the neuron doesn't fire, so zero
    # them out. This doesn't zero out the node's in-degree, so it might be
    # activated on a subsequent iteration.
    activations[activations <= update.x[1]] = 0

    if debug:
        print(f"new_adj, before zeroing: {new_adj}")

    # If a node is activated above threshold (i.e., if a neuron has fired),
    # zero out its in-degree in future passes to eliminate loops, simulating
    # a refractory period.
    #
    # N.B.: It's not obvious but ">=" and "> 0" are both wrong here, and this
    # comparison is matched to the comparison used when zeroing
    # downstream_activations. The comparisons need to cover update()'s domain.
    new_adj[:, activations > update.x[1]] = 0
    if debug:
        print(f"new_adj, after zeroing: {new_adj}")

    downstream_activations = np.clip(np.dot(new_adj.T, activations) * discount, 0, 1)
    if debug:
        print("downstream_activations, before zeroing: " f"{downstream_activations}")

    # If an activation is below the threshold where it would make a difference,
    # zero it out.
    #
    # N.B.: The "<=" here is matched to the ">" above used for zeroing new_adj.
    downstream_activations[downstream_activations <= update.x[1]] = 0

    if debug:
        print("downstream_activations, after zeroing: " f"{downstream_activations}")

    if debug:
        print(f"update.x: {update.x}")
        print(f"downstream_acts: {downstream_activations}")

    if np.all(downstream_activations == 0):
        return activations
    else:
        return activations + spread_recur(
            adj,
            downstream_activations,
            update,
            discount=discount,
            new_adj=new_adj,
            debug=debug,
        )


def update_recur(adj, activations, update, debug=False):
    """Reweight edges according to a non-monotonic plasticity function.

    Args:
        adj (np.array, 2D): Weighted adjacency matrix.
        activations (np.array, 1D): Activation strengths for each node.
        update (function): update function for non-monotonic reweighting.
        debug (bool): Print debug information.
    """
    # for each node, change all of its input weights according to the update
    # modify both weights according to the minimum activation of the nodes
    if debug:
        print(f"input activations: {activations}")

    act_mesh = np.meshgrid(activations, activations)
    min_acts = np.minimum(act_mesh[0], act_mesh[1])
    try:
        adjust = update(min_acts)
    except Exception as e:
        print("problem applying update function")
        min_acts_list = list(min_acts)
        print("min_acts:")
        for item in min_acts_list:
            print(item)
        print(f"update.x: {update.x}")
        print(f"update.y: {update.y}")
        print("Exception:")
        print(e)
    new_adj = np.clip(adj + adjust, 0, 1)
    np.fill_diagonal(new_adj, 0)
    return new_adj


def compute_adjacency(W):
    """Given a weight matrix W, return an adjacency matrix"""
    A = W.copy()
    A[A > 0.0] = 1.0
    return A
