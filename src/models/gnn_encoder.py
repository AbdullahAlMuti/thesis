"""
Dynamic Graph Neural Network (PyTorch Geometric GATv2Conv) for Cross-Instrument Causal Embedding.
Consumes temporal market graph G_t = (V, E, A_t) and outputs dense causal state representations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Tuple, Dict, Any, Optional

class MarketGNNEncoder(nn.Module):
    """
    Graph Attention Network (GATv2) for Market Graph Embedding.
    """

    def __init__(self, in_channels: int = 8, hidden_channels: int = 32, out_channels: int = 16, heads: int = 2):
        super(MarketGNNEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads

        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.fc_embedding = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass:
        x: Node feature matrix [N, in_channels]
        edge_index: Graph edge indices [2, E]
        edge_attr: Edge weights/causal correlation [E, 1]
        Returns graph-level node embeddings [N, out_channels]
        """
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.conv2(h, edge_index)
        out = self.fc_embedding(h)
        return out

    def extract_target_embedding(self, x: torch.Tensor, edge_index: torch.Tensor, target_idx: int = 0) -> torch.Tensor:
        """
        Extracts embedding vector specifically for EURUSD (target node index 0).
        """
        node_embeddings = self.forward(x, edge_index)
        return node_embeddings[target_idx]
