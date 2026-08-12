import torch
import torch.nn as nn

class EmRegulizer(nn.Module):
    def __init__(
        self,
        xi: float = 0.0,
        lambda_var: float = 0.0,
        eps: float = 1e-8,
        assume_symmetric: bool = True,
    ):
        super().__init__()

        if xi < 0:
            raise ValueError("xi must be non-negative.")

        if lambda_var < 0:
            raise ValueError(
                "lambda_var must be non-negative."
            )

        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.xi = xi
        self.lambda_var = lambda_var
        self.eps = eps
        self.assume_symmetric = assume_symmetric

    def forward(
        self,
        f: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        
        if f.dim() != 1:
            raise ValueError(
                "f must have shape [num_nodes]. "
                f"Got {tuple(f.shape)}."
            )

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                "edge_index must have shape [2, num_edges]."
            )

        row, col = edge_index

        diff = f[row] - f[col]

        smoothness = diff.square().sum()

        if self.assume_symmetric:
            smoothness = 0.5 * smoothness


        variance = torch.var(
            f,
            unbiased=False,
        )

        variance_penalty = (
            1.0
            / variance.clamp_min(self.eps)
        )

        loss = (
            self.xi * smoothness
            + self.lambda_var * variance_penalty
        )

        return loss

    def components(
        self,
        f: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        row, col = edge_index

        diff = f[row] - f[col]

        smoothness = diff.square().sum()

        if self.assume_symmetric:
            smoothness = 0.5 * smoothness

        variance = torch.var(
            f,
            unbiased=False,
        )

        variance_penalty = (
            1.0
            / variance.clamp_min(self.eps)
        )

        total = (
            self.xi * smoothness
            + self.lambda_var * variance_penalty
        )

        return (
            smoothness,
            variance,
            variance_penalty,
            total,
        )