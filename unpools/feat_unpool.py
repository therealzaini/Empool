import torch
import torch.nn as nn

class XUnpool(nn.Module):

    def forward(
            self,
            x_coarse: torch.Tensor,
            sink_index: torch.Tensor
    ) -> torch.Tensor:
        if x_coarse.dim() != 2:
            raise ValueError("x_coarse dimension mismatch.")
        if sink_index.dim() != 1:
            raise ValueError("sink_index dimension mismatch.")
        
        if sink_index.numel() == 0:
            return x_coarse.new_empty(
                (0, x_coarse.size(1))
            )

        if sink_index.min() < 0:
            raise ValueError("sink_index contains negative values.")

        if sink_index.max() >= x_coarse.size(0):
            raise ValueError("sink_index refers to a coarse node outside the range of x_coarse.")

        return x_coarse[sink_index]