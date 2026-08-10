import torch
import torch.nn as nn

from typing import List

from .decoder_block import DecoderBlock
from ..encoder.encoder_state import EncoderState

class MultiDecoder(nn.Module):
    def __init__(
            self,
            encoder_dim: List[int],
            decoder_dim: List[int],
            layer_type: str,
            heads: int,
            dropout: float
    ):
        super().__init__()

        self.encoders_dim = encoder_dim
        num_encoding_steps = len(self.encoders_dim)

        self.decoders_dim = decoder_dim

        blocks = []

        for decoder_index in range(num_encoding_steps):

            encoder_index = num_encoding_steps - 1 - decoder_index

            coarse_channels = self.encoders_dim[encoder_index] if decoder_index == 0 else decoder_dim[decoder_index - 1]

            skip_channels = encoder_dim[encoder_index]
            out_channels = decoder_dim[decoder_index]

            blocks.append(
                DecoderBlock(
                    coarse_dimension = coarse_channels,
                    skip_dimension = skip_channels,
                    out_channels = out_channels,
                    layer_type = layer_type,
                    heads = heads,
                    dropout = dropout
                )
            )

        self.blocks = nn.ModuleList(blocks)


    def forward(
            self,
            x_bott: torch.Tensor,
            state_stack: List[EncoderState]
    ) -> torch.Tensor:

        x = x_bott

        for block in self.blocks:
            state = state_stack.pop()

            x = block(
                x_coarse = x,
                edge_index = state.edge_index,
                sink_index = state.sink_index,
                x_skip = state.x_skip
            )

        return x