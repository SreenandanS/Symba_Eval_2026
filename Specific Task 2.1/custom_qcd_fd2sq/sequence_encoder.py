"""
Sequence encoder baseline for custom-encoder ablations.

This encoder consumes a serialized diagram sequence rather than a graph.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .encoder import CrossStreamAttention, GraphReadout
from .sequence_dataset import NUM_SEGMENT_TYPES, SEGMENT_PROPAGATOR


class SinusoidalPosEmb(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


class SequenceDiagramEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        hidden_dim: int = 128,
        stream_dim: int = 64,
        graph_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim
        self.stream_dim = stream_dim

        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.segment_emb = nn.Embedding(NUM_SEGMENT_TYPES, hidden_dim)
        self.pos_emb = SinusoidalPosEmb(hidden_dim, max_len=max_seq_len * 2)
        self.emb_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=max(4 * hidden_dim, 256),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.kinematic_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stream_dim),
        )
        self.color_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stream_dim),
        )
        self.spinor_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stream_dim),
        )
        self.fusion = CrossStreamAttention(
            stream_dim=stream_dim,
            num_streams=3,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.readout = GraphReadout(
            node_dim=stream_dim,
            out_dim=graph_dim,
            dropout=dropout,
        )

    def _encode_dense(self, batch) -> tuple[Tensor, Tensor, Tensor]:
        src_ids = batch.src_ids
        src_segment_ids = batch.src_segment_ids
        mask = src_ids != self.pad_id

        x = self.token_emb(src_ids) * math.sqrt(self.hidden_dim)
        x = x + self.segment_emb(src_segment_ids)
        x = self.pos_emb(x)
        x = self.emb_drop(x)
        x = self.encoder(
            x,
            src_key_padding_mask=~mask,
        )
        return x, mask, src_segment_ids

    def encode_context(self, batch) -> dict[str, Tensor]:
        encoded, mask, segment_ids = self._encode_dense(batch)
        batch_size, seq_len, _ = encoded.shape
        batch_index = torch.arange(
            batch_size,
            device=encoded.device,
        ).unsqueeze(1).expand(batch_size, seq_len)

        kinematic_dense = self.kinematic_proj(encoded)
        color_dense = self.color_proj(encoded)
        spinor_dense = self.spinor_proj(encoded)

        kinematic_flat = kinematic_dense[mask]
        color_flat = color_dense[mask]
        spinor_flat = spinor_dense[mask]
        fused_flat = self.fusion(kinematic_flat, color_flat, spinor_flat)

        memory = encoded.new_zeros(
            batch_size,
            seq_len,
            self.stream_dim,
        )
        memory[mask] = fused_flat

        return {
            "memory": memory,
            "memory_mask": mask,
            "batch": batch_index[mask],
            "kinematic": kinematic_flat,
            "color": color_flat,
            "spinor": spinor_flat,
            "fused": fused_flat,
            "propagator_mask": (segment_ids[mask] == SEGMENT_PROPAGATOR),
        }

    def forward(self, batch) -> Tensor:
        encoded = self.encode_context(batch)
        return self.readout(encoded["fused"], encoded["batch"])

    def get_stream_embeddings(self, batch) -> dict[str, Tensor]:
        encoded = self.encode_context(batch)
        return {
            "kinematic": encoded["kinematic"],
            "lorentz": encoded["kinematic"],
            "color": encoded["color"],
            "spinor": encoded["spinor"],
            "fused": encoded["fused"],
        }
