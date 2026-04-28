"""GRU rescorer configuration."""

from __future__ import annotations

from pydantic import BaseModel


class GRUConfig(BaseModel):
    input_dim: int = 768
    hidden_dim: int = 128
    num_layers: int = 1
    bidirectional: bool = True
    dropout_input: float = 0.3

    # Training.
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 16

    # Volume score top-k.
    top_k: int = 5
    aux_loss_weight: float = 0.1
