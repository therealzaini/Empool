import torch
import torch.nn as nn

from typing import List

from encoder.encoder_block import EncoderBlock
from encoder.encoder_state import EncoderState
from utils.topology_stats import TopologyStats

class MultiEncoder(nn.Module):

    def __init__(
            self,
            in_channels: int,
            hidden_channels: List[int],
            num_layers: int,
            layer_type: str = "gcn",
            heads: int = 1,
            dropout: float = 0.0,
            initializer_alpha: float = 0.01,
            initializer_iterations: int = 10,
            backtracking_iterations: int = 1,
            surrogate_tau: float = 1.0,
            pooling_eps: float = 1e-8,
    ):
        super().__init__()
        
        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if len(hidden_channels) != num_layers:
            raise ValueError(
                "hidden_channels must contain exactly "
                "num_layers entries."
            )

        self.num_layers = num_layers

        encoder_blocks = []

        current = in_channels

        for layer in range(num_layers):
            nex = hidden_channels[layer]

            enc_block = EncoderBlock(
                in_channels = current,
                out_channels = nex,
                layer_type = layer_type,
                heads = heads,
                dropout = dropout,
                initializer_alpha = initializer_alpha,
                initializer_iterations = initializer_iterations,
                backtracking_iterations = backtracking_iterations,
                surrogate_temperature = surrogate_tau,
                pooling_eps = pooling_eps
            )

            encoder_blocks.append(enc_block)
            current = nex

        self.blocks = nn.ModuleList(encoder_blocks)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor
    ):
        state_stack = []

        current_x, current_edge_index = x, edge_index


        num_input_nodes = current_x.size(0)
        node_count_per_layer = []
        sink_count_per_layer = []
        sink_ratios = []

        for block in self.blocks:

            node_count = current_x.size(0)

            current_x, current_edge_index, sink_index, num_sinks, x_skip = block(current_x, current_edge_index)

            ratio = num_sinks / node_count

            node_count_per_layer.append(node_count)
            sink_count_per_layer.append(num_sinks)
            sink_ratios.append(ratio)

            state_stack.append(EncoderState(x_skip, current_edge_index, sink_index, num_sinks))

        bottlneck_node_count = current_x.size(0)

        global_sink_ratio = bottlneck_node_count / num_input_nodes

        stats = TopologyStats(node_count_per_layer, sink_count_per_layer, sink_ratios, global_sink_ratio)

        return current_x, current_edge_index, state_stack, stats
