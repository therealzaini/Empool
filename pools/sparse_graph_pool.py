import torch
import torch.nn as nn
from torch_geometric.utils import coalesce

class SparseGPool(nn.Module):

    def __init__(self, remove_self_loops: bool = True):
        super().__init__()
        self.remove_self_loops = remove_self_loops

    def forward(
            self,
            edge_index: torch.Tensor,
            sink_index: torch.Tensor,
            num_sinks: int
    ) -> torch.Tensor:


        row, col = edge_index

        row_pool, col_pool = sink_index[row], sink_index[col]

        if self.remove_self_loops:
            mask = row_pool != col_pool

            row_pool = row_pool[mask]
            col_pool = col_pool[mask]

        pooled_edge_index = torch.stack([row_pool, col_pool], dim = 0)

        pooled_edge_index = coalesce(pooled_edge_index, num_nodes = num_sinks)

        return pooled_edge_index
