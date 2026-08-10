import torch
import torch.nn as nn

class MessagePassing(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            layer_type: str,
            heads: int,
            dropout: float,
            **kwargs
    ):
        super().__init__()

        layer_type = layer_type.lower()

        if layer_type == 'gcn':
            from torch_geometric.nn import GCNConv
            self.conv = GCNConv(in_channels, out_channels, **kwargs)

        elif layer_type == 'gat':
            from torch_geometric.nn import GATConv
            self.conv = GATConv(in_channels, out_channels, heads, dropout = dropout, **kwargs)

        else:
            raise ValueError(f"{layer_type} layer type invalid or unimplemented.")

        self.norm = nn.LayerNorm(out_channels * heads if layer_type == 'gat' else out_channels)

        self.activation = nn.ELU()


    def forward(
            self,
            X,
            edge_index
    ):
        X = self.conv(X, edge_index)
        X = self.norm(X)
        X = self.activation(X)

        return X