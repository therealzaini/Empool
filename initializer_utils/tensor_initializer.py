import torch
from torch import Tensor

class TensorInitializer:

    def __init__(
            self,
            alpha: float,
            num_iterations: int
    ):

        if alpha <= 0:
            raise ValueError("alpha must be strictly positive.")
        if num_iterations < 0:
            raise ValueError("num_iterations must be non-negative")

        self.alpha = alpha
        self.num_iter = num_iterations


    @staticmethod
    def _validate_alpha_and_get_degree(
        alpha: float,
        edge_index: Tensor,
        num_nodes: int
    ):
        row = edge_index[0]

        degree = torch.bincount(
            row,
            minlength = num_nodes
        )

        d_max = int(degree.max().item())

        if d_max == 0:
            return degree

        upper_bound = 1.0 / (2.0 * d_max)

        if alpha >= upper_bound:
            raise ValueError(f"Value of alpha out of bounds. Choose a strictly positive value no bigger than 1/2d_max = {upper_bound}.")

        return degree

    def initialize(
        self,
        edge_index: torch.Tensor,
        reference: torch.Tensor,
    ):
        num_nodes = reference.size(0)
        device = reference.device
        dtype = reference.dtype

        # Degrees are integer-valued, but are only used
        # to determine d_max.
        row = edge_index[0]

        degree = torch.bincount(
            row,
            minlength=num_nodes,
        )

        d_max = degree.max().item()

        # Random Gaussian seed
        x_seed = torch.randn(
            num_nodes,
            device=device,
            dtype=dtype,
        )

        # Center
        x_seed = x_seed - x_seed.mean()

        x_seed = x_seed / x_seed.norm(p=2).clamp_min(torch.finfo(dtype).eps)

        row, col = edge_index

        for _ in range(self.num_iter):

            Ax = torch.zeros_like(x_seed)
            Ax.scatter_add_(
                0,
                row,
                x_seed[col]
            )

            Dx = degree * x_seed

            x_seed = x_seed - self.alpha * (Dx - Ax)

            x_seed = x_seed / x_seed.norm(p = 2).clamp_min(torch.finfo(dtype).eps)

        return x_seed