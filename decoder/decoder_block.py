import torch
import torch.nn as nn

from unpools.feat_unpool import XUnpool
from message_passing.message_passing import MessagePassing

class DecoderBlock(nn.Module):
    def __init__(
                self,
                coarse_dimension: int,
                skip_dimension: int,
                out_channels: int,
                layer_type: str,
                heads: int,
                dropout: float
        ):
            super().__init__()

            self.coarse = coarse_dimension
            self.skip = skip_dimension
            self.out_channels = out_channels

            self.unpool = XUnpool()

            self.message_passing = MessagePassing(
                in_channels = coarse_dimension + skip_dimension,
                out_channels = out_channels,
                layer_type = layer_type,
                heads = heads,
                dropout = dropout,
            )

    def forward(
                self,
                x_coarse: torch.Tensor,
                edge_index: torch.Tensor,
                sink_index: torch.Tensor,
                x_skip: torch.Tensor
    ) -> torch.Tensor:

        x_unpooled = self.unpool(x_coarse, sink_index)

        if x_unpooled.size(0) != x_skip.size(0):
             raise RuntimeError("Dimension mismatch: cannot concatenate features.")

        x = torch.cat(
             [x_unpooled, x_skip],
             dim = -1
        )

        x = self.message_passing(x, edge_index)

        return x