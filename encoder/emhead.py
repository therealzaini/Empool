import torch
import torch.nn as nn

from typing import Optional, cast

class Emhead(nn.Module):

    def __init__(
            self,
            in_channels: int,
            hidden_channels: Optional[int],
            dropout: float = 0.0
    ):

        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        if hidden_channels is None:
            hidden_channels = max(1, in_channels // 2)

        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.nn = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(hidden_channels, 1)
        )

        minus_oneth_layer = self.nn[-1]

        nn.init.zeros_(cast(torch.Tensor, minus_oneth_layer.weight))
        nn.init.zeros_(cast(torch.Tensor, minus_oneth_layer.bias))


    def forward(
            self,
            X: torch.Tensor,
            f_init: torch.Tensor
    ) -> torch.Tensor:

        if X.dim() != 2:
            raise ValueError(f"X must have shape [num_nodes, num_features]. Got shape {tuple(X.shape)}.")

        if f_init.dim() != 1:
            raise ValueError(f"X must have shape [num_nodes]. Got shape {tuple(f_init.shape)}.")

        if X.size(0) != f_init.numel():
            raise ValueError("Dimension mismatch between features and Morse vector.")

        fluctuations = self.nn(X).squeeze(-1)

        f = f_init + fluctuations

        return f