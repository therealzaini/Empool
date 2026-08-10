from typing import Tuple

import torch
from torch_scatter import scatter_max

def construct_sink(
        edge_index: torch.Tensor,
        num_nodes: int,
        morse_values: torch.Tensor,
        max_iterations: int = 4
) -> Tuple[torch.Tensor, int]:
    
    """
    This function implements Algorithm 1 from our paper. 

    Args:
        edge_index (torch.Tensor): The tensor representing edge indices. This argument is typically passed as 
                ``data.edge_index``.
        num_nodes (int): Self-explanatory. This argument is typically passed as ``data.num_nodes``.
        morse_values (torch.Tensor): This argument represents the Morse vector governing the Forman-Lefschetz collapse
                algorithm, in the form of a 1D tensor of length ``num_nodes`` such that ``morse_values[i]``
                represents the Morse value of vertex of index ``i``.
        max_iterations (int): The maximum number of iterations to resolve nodes parents. Set to 4 by default.

    Returns:
        Tuple[torch.Tensor, int]: a 1D tensor of length ``num_nodes`` such that the i-th entry represents the index 
                of the critical node the i-th vertex sank into, and an integer representing the number of critical nodes.

    Raises:
        ValueError: if ``morse_values`` has an invalid format or ``max_iterations`` is strictly negative.
    """

    device = edge_index.device

    if morse_values.ndim != 1 or morse_values.numel() != num_nodes:
        raise ValueError("morse_values must be a 1D tensor with one value per node.")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative.")

    source_node, destination_node = edge_index[0], edge_index[1]

    delta_f = morse_values[source_node] - morse_values[destination_node]

    max_delta_f, negative_nabla_edge_index = scatter_max(
         delta_f,
         source_node,
         dim = 0,
         dim_size = num_nodes
    )

    if edge_index.numel() == 0:
        return torch.arange(num_nodes, device=device, dtype=torch.long), num_nodes

    negative_nabla_edge_index = torch.clamp(
         negative_nabla_edge_index,
         min = 0,
         max = source_node.numel() - 1
    )

    negative_nabla = destination_node[negative_nabla_edge_index]
    negative_nabla = torch.clamp(negative_nabla, 0, num_nodes - 1)

    is_critical = (max_delta_f <= 1e-5) | (negative_nabla == torch.arange(num_nodes, device = device))

    parent_array = torch.where(
         is_critical,
         torch.arange(num_nodes, device = device, dtype = torch.long),
         negative_nabla
    )

    for _ in range(max_iterations):
        parent_array = parent_array[parent_array]

    parent_array = torch.clamp(parent_array, 0, num_nodes - 1)

    critical_nodes = torch.nonzero(is_critical, as_tuple = False).flatten()
    critical_nodes = critical_nodes[(critical_nodes >= 0) & (critical_nodes < num_nodes)]

    number_of_sinks = critical_nodes.numel()

    if number_of_sinks == 0:
        return torch.arange(num_nodes, device=device, dtype=torch.long), 0

    node_to_sink = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    node_to_sink[critical_nodes] = torch.arange(number_of_sinks, device=device)

    sink_index = node_to_sink[parent_array]

    unmapped_nodes = (sink_index < 0) | (sink_index >= number_of_sinks)

    if unmapped_nodes.any():
        sink_index = torch.where(
            unmapped_nodes, 
            torch.arange(num_nodes, device=device, dtype=torch.long) % number_of_sinks, 
            sink_index
        )

    return sink_index, number_of_sinks    