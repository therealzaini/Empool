import torch
import torch.nn as nn

class LocalizedHeatDissipationSurrogate(nn.Module):

    def __init__(
            self,
            tau: float = 1.0,
            max_candidates: int = 8
    ):
        
        super().__init__()

        if tau <= 0:
            raise ValueError("tau must be strictly positive.")

        if max_candidates < 0:
            raise ValueError("max_candidates must be positive.")

        self.tau = tau
        self.max_candidates = max_candidates

    @staticmethod
    def _build_candidates(
        edge_index: torch.Tensor,
        sink_index: torch.Tensor,
        num_nodes: int
    ):

        row, col = edge_index
        device = sink_index.device

        own_node = torch.arange(num_nodes, device = device, dtype = torch.long)

        own_sink = sink_index

        neighbor_node = torch.cat(
            [row, col],
            dim = 0
        )

        neighbor_sink = torch.cat(
            [sink_index[col], sink_index[row]],
            dim = 0
        )

        node = torch.cat(
            [own_node, neighbor_node],
            dim = 0
        )

        candidate = torch.cat(
            [own_sink, neighbor_sink]
        )

        number_of_sinks = int(sink_index.max().item()) + 1

        pair_code = (node * number_of_sinks + candidate)

        pair_code = torch.unique(pair_code, sorted = False)

        node = torch.div(
            pair_code,
            number_of_sinks,
            rounding_mode = "floor"
        )

        candidate = (pair_code % number_of_sinks)

        return node, candidate

    def forward(
            self,
            edge_index: torch.Tensor,
            morse_values: torch.Tensor,
            sink_index: torch.Tensor
    ):

        device = morse_values.device
        num_nodes = morse_values.numel()
        dtype = morse_values.dtype

        node, candidate = self._build_candidates(
            edge_index = edge_index,
            sink_index = sink_index,
            num_nodes = num_nodes
        )

        score = -(morse_values[node] - morse_values[candidate]).pow(2)

        max_score = torch.full(
            (num_nodes,),
            -torch.inf,
            device = device,
            dtype = dtype
        )

        max_score.scatter_reduce_(0, node, score, reduce = "amax", include_self = True)
        e_to_the_score = torch.exp((score - max_score[node]) / self.tau)

        sum_of_exps = torch.zeros(num_nodes, device = device, dtype = dtype)

        sum_of_exps.scatter_add_(0, node, e_to_the_score)

        soft_p = (e_to_the_score / sum_of_exps[node].clamp_min(torch.finfo(dtype).eps))
        hard_p = (candidate == sink_index[node]).to(device)

        return node, candidate, soft_p, hard_p
