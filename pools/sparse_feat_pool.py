import torch
import torch.nn as nn

class SparseXPool(nn.Module):

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.epsilon = eps

    def forward(
            self,
            X: torch.Tensor,
            node: torch.Tensor,
            candidate: torch.Tensor,
            weight: torch.Tensor,
            num_sinks: int
    ) -> torch.Tensor:

        device, dtype = X.device, X.dtype

        weighted_X = (X[node] * weight.unsqueeze(-1))

        X_pool = torch.zeros(num_sinks, X.size(-1), device = device, dtype = dtype)

        sink_index = candidate.unsqueeze(-1).expand_as(weighted_X)

        X_pool.scatter_add_(0, sink_index, weighted_X)

        D_pool = torch.zeros(num_sinks, device = device, dtype = dtype)

        D_pool.scatter_add_(0, candidate, weight)

        X_pool = X_pool / D_pool.clamp_min(self.epsilon).unsqueeze(-1)

        return X_pool