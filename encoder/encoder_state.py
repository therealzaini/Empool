from dataclasses import dataclass
import torch 
import torch.nn as nn

@dataclass
class EncoderState:
    x_skip: torch.Tensor
    edge_index: torch.Tensor
    sink_index: torch.Tensor
    num_sinks: int