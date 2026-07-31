"""
Unit tests for MarketGNNEncoder (PyTorch Geometric GATv2).
"""
import torch
import pytest
from src.models.gnn_encoder import MarketGNNEncoder

def test_gnn_encoder_forward_pass():
    # 6 nodes (EURUSD, GBPUSD, USDJPY, USDCHF, XAUUSD, US500), 8 features each
    x = torch.randn(6, 8)
    # Fully connected edge index
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0],
        [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=torch.long)

    model = MarketGNNEncoder(in_channels=8, hidden_channels=16, out_channels=8, heads=2)
    out = model(x, edge_index)

    assert out.shape == (6, 8)

def test_extract_target_embedding():
    x = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    model = MarketGNNEncoder(in_channels=8, hidden_channels=16, out_channels=8, heads=2)
    target_emb = model.extract_target_embedding(x, edge_index, target_idx=0)

    assert target_emb.shape == (8,)
