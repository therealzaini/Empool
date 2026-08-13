import torch
import torch.nn as nn

from typing import Optional, cast

from initializer_utils.tensor_initializer import TensorInitializer

class Emhead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        dropout: float = 0.0,
        initializer: TensorInitializer | None = None,
    ):
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be positive."
            )

        if hidden_channels is None:
            hidden_channels = max(
                1,
                in_channels // 2,
            )

        if hidden_channels <= 0:
            raise ValueError(
                "hidden_channels must be positive."
            )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # ---------------------------------------------------------
        # First MLP layer.
        # ---------------------------------------------------------

        self.input_layer = nn.Linear(
            in_channels,
            hidden_channels,
        )

        self.activation = nn.ELU()

        self.dropout = nn.Dropout(
            p=dropout
        )

        # ---------------------------------------------------------
        # W2 is deliberately NOT nn.Linear(hidden, 1).
        #
        # We need a separately controlled W2 whose initial value
        # is exactly zero.
        #
        # Shape:
        #
        #     [1, hidden_channels]
        # ---------------------------------------------------------

        self.output_weight = nn.Parameter(
            torch.zeros(
                1,
                hidden_channels,
            )
        )

        # ---------------------------------------------------------
        # Cached node-wise b2.
        #
        # It is a BUFFER, not a learnable parameter.
        #
        # b2[l] corresponds to the Morse initialization at the
        # current encoder level.
        # ---------------------------------------------------------

        self.register_buffer(
            "_b2",
            None,
            persistent=True,
        )

        # Keep initializer as a component of the head.
        self.initializer = initializer

    @property
    def b2(self):
        """Return the currently cached Morse offset."""
        return self._b2

    def _initialize_b2(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Generate and cache the node-wise Morse initialization.

        IMPORTANT:
        This is called only when there is no cached b2 or when
        the current graph requires a different number of entries.
        """

        if self.initializer is None:
            # Explicit fallback for experiments where we want
            # zero initialization.
            b2 = torch.zeros(
                num_nodes,
                device=device,
                dtype=dtype,
            )

        else:
            b2 = self.initializer.initialize(
                edge_index=edge_index,
                num_nodes=num_nodes,
                device=device,
                dtype=dtype,
            )

        # Store as a fixed buffer.
        self._b2 = b2.detach()

    def _ensure_b2(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Ensure that a compatible cached initialization exists.
        """

        needs_initialization = (
            self._b2 is None
            or self._b2.numel() != num_nodes
            or self._b2.device != device
            or self._b2.dtype != dtype
        )

        if needs_initialization:
            self._initialize_b2(
                edge_index=edge_index,
                num_nodes=num_nodes,
                device=device,
                dtype=dtype,
            )

    def forward(
        self,
        H: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        H:
            Node representation [N, F].

        edge_index:
            Current graph connectivity [2, E].

        Returns
        -------
        f:
            Morse vector [N].

        Computes:

            f = W2 phi(W1 H + b1) + b2
        """

        if H.dim() != 2:
            raise ValueError(
                "H must have shape [num_nodes, num_features]."
            )

        num_nodes = H.size(0)

        # ---------------------------------------------------------
        # Make sure b2 exists and matches this graph's resolution.
        # ---------------------------------------------------------

        self._ensure_b2(
            edge_index=edge_index,
            num_nodes=num_nodes,
            device=H.device,
            dtype=H.dtype,
        )

        # ---------------------------------------------------------
        # Learned hidden representation.
        # ---------------------------------------------------------

        hidden = self.input_layer(H)

        hidden = self.activation(hidden)

        hidden = self.dropout(hidden)

        # ---------------------------------------------------------
        # Learned Morse correction.
        #
        # W2 initially equals zero, so this is initially zero.
        # ---------------------------------------------------------

        correction = (
            hidden
            @ self.output_weight.t()
        ).squeeze(-1)

        # ---------------------------------------------------------
        # Final Morse vector.
        # ---------------------------------------------------------

        f = correction + self._b2

        return f