from typing import Tuple

import torch
from torch_scatter import scatter_max


def construct_sink(
    edge_index: torch.Tensor,
    num_nodes: int,
    morse_values: torch.Tensor,
    max_iterations: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Construct truncated Forman-collapse sink assignments.

    Parameters
    ----------
    edge_index:
        Graph connectivity [2, E].

    num_nodes:
        Number of vertices.

    morse_values:
        Morse function f(v) for every vertex [N].

    max_iterations:
        Number T of pointer-jumping iterations performed after
        the initial negative-gradient assignment.

    Returns
    -------
    sink_index:
        Compact sink assignment [N].

        sink_index[v] = k means that vertex v is represented by
        sink k.

    sink_nodes:
        Actual vertex represented by each sink [K].

        sink_nodes[k] = representative vertex of sink k.

    num_sinks:
        Number K of T-sink representatives.
    """

    device = edge_index.device

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if morse_values.ndim != 1:
        raise ValueError(
            "morse_values must be a 1D tensor."
        )

    if morse_values.numel() != num_nodes:
        raise ValueError(
            "morse_values must contain exactly one value "
            "per node."
        )

    if max_iterations < 0:
        raise ValueError(
            "max_iterations must be non-negative."
        )

    # ---------------------------------------------------------
    # Empty graph
    # ---------------------------------------------------------

    if edge_index.numel() == 0:

        # Every isolated vertex is a local minimum and therefore
        # its own representative.
        sink_nodes = torch.arange(
            num_nodes,
            device=device,
            dtype=torch.long,
        )

        sink_index = sink_nodes.clone()

        return (
            sink_index,
            sink_nodes,
            num_nodes,
        )

    source = edge_index[0]
    destination = edge_index[1]

    # ---------------------------------------------------------
    # Initial negative gradient.
    #
    # For every source vertex v, find the neighbor u maximizing
    #
    #     f(v) - f(u)
    #
    # which is equivalent to maximizing the downhill descent.
    # ---------------------------------------------------------

    delta_f = (
        morse_values[source]
        - morse_values[destination]
    )

    max_delta_f, max_edge = scatter_max(
        delta_f,
        source,
        dim=0,
        dim_size=num_nodes,
    )

    # scatter_max returns -inf / invalid index for nodes with
    # no outgoing edge. The usual Cora/PyG graph is represented
    # symmetrically, but handle the case explicitly.
    valid = torch.isfinite(max_delta_f)

    max_edge = torch.where(
        valid,
        max_edge,
        torch.zeros_like(max_edge),
    )

    negative_gradient = destination[max_edge]

    # ---------------------------------------------------------
    # Local minima / isolated vertices.
    #
    # A vertex is initially critical when there is no strictly
    # downhill neighbor.
    # ---------------------------------------------------------

    critical = (
        (~valid)
        | (max_delta_f <= 0)
    )

    vertex_ids = torch.arange(
        num_nodes,
        device=device,
        dtype=torch.long,
    )

    # ---------------------------------------------------------
    # Initial parent assignment.
    #
    # Critical vertex:
    #       pi(v) = v
    #
    # Non-critical vertex:
    #       pi(v) = negative_gradient(v)
    # ---------------------------------------------------------

    parent = torch.where(
        critical,
        vertex_ids,
        negative_gradient,
    )

    # ---------------------------------------------------------
    # Bounded pointer jumping.
    #
    # pi <- pi o pi
    #
    # performed exactly T times.
    # ---------------------------------------------------------

    for _ in range(max_iterations):
        parent = parent[parent]

    # ---------------------------------------------------------
    # Define T-sinks as the distinct representatives remaining
    # after the bounded pointer-jumping process.
    #
    # sink_nodes contains ACTUAL VERTEX IDS.
    # ---------------------------------------------------------

    sink_nodes, sink_index = torch.unique(
        parent,
        sorted=True,
        return_inverse=True,
    )

    num_sinks = sink_nodes.numel()

    return (
        sink_index,
        sink_nodes,
        num_sinks,
    )