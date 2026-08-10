import torch
import torch.nn as nn

from typing import List

from decoder.multidecoder import MultiDecoder
from encoder.multiencoder import MultiEncoder

class Empool(nn.Module):

    def __init__(
        self,
        in_channels: int,
        encoder_channels: List[int],
        decoder_channels: List[int],
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

        num_layers = len(encoder_channels)

        if len(decoder_channels) != num_layers:
            raise ValueError(
                "decoder_channels must have the same length "
                "as encoder_channels."
            )

        self.encoder = MultiEncoder(
            in_channels=in_channels,
            hidden_channels=encoder_channels,
            num_layers=num_layers,
            layer_type=layer_type,
            heads=heads,
            dropout=dropout,
            initializer_alpha=initializer_alpha,
            initializer_iterations=initializer_iterations,
            backtracking_iterations=backtracking_iterations,
            surrogate_tau=surrogate_tau,
            pooling_eps=pooling_eps,
        )

        self.bottleneck = nn.Identity()

        self.decoder = MultiDecoder(
            encoder_dim=encoder_channels,
            decoder_dim=decoder_channels,
            layer_type=layer_type,
            heads=heads,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_stats: bool = False,
    ):

        x, edge_index, state_stack, stats = self.encoder(
            x,
            edge_index,
        )

        x = self.bottleneck(x)

        x = self.decoder(
            x_bott=x,
            state_stack=state_stack,
        )

        if return_stats:
            return x, stats

        return x