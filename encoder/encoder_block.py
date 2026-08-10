import torch
import torch.nn as nn

from message_passing.message_passing import MessagePassing
from initializer_utils.tensor_initializer import TensorInitializer
from encoder.emhead import Emhead
from sink_utils.sink_ste import LocalizedHeatDissipationSurrogate as STE
from pools.sparse_feat_pool import SparseXPool
from pools.sparse_graph_pool import SparseGPool
from sink_utils.sink import construct_sink

class EncoderBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            layer_type: str,
            heads: int,
            dropout: float,
            initializer_alpha: float = 1e-3,
            initializer_iterations: int = 5,
            backtracking_iterations: int = 1,
            surrogate_temperature: float = 1.0,
            pooling_eps: float = 1e-8
    ):
        super().__init__()

        self.message_passing = MessagePassing(
            in_channels = in_channels,
            out_channels = out_channels,
            layer_type = layer_type,
            heads = heads,
            dropout = dropout
        )

        self.initializer = TensorInitializer(initializer_alpha, initializer_iterations)


        self.emhead = Emhead(in_channels = out_channels, hidden_channels = max(1, out_channels // 2))

        self.ste = STE(tau = surrogate_temperature)

        self.Xpool = SparseXPool(eps = pooling_eps)
        self.Gpool = SparseGPool(True)

        self.backtracking_iter = backtracking_iterations


    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor
    ):
        

        H = self.message_passing(x, edge_index)

        x_skip = H

        f_init = self.initializer.initialize(edge_index = edge_index, reference = H)

        f = self.emhead(H, f_init)

        sink_index, num_sinks = construct_sink(edge_index, x.size(0), f, self.backtracking_iter)

        node, candidate, soft_p, hard_p = self.ste(edge_index, f, sink_index)

        SINK = hard_p + soft_p - soft_p.detach()

        X_pool = self.Xpool(H, node, candidate, SINK, num_sinks)
        pooled_edge_index = self.Gpool(edge_index, sink_index, num_sinks)

        return X_pool, pooled_edge_index, sink_index, num_sinks, x_skip


        