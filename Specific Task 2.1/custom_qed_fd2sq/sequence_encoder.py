"""Sequence encoder baseline aligned to the canonical QED slot contract."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .features import STATIC_CHARGE_FEATURE_DIM
from .sequence_dataset import NUM_SEGMENT_TYPES


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
        slot_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        static_charge_feature_dim: int = STATIC_CHARGE_FEATURE_DIM,
        **_: int,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim
        self.slot_dim = slot_dim

        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.segment_emb = nn.Embedding(NUM_SEGMENT_TYPES, hidden_dim)
        self.pos_emb = SinusoidalPosEmb(hidden_dim, max_len=max_seq_len * 2)
        self.emb_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=max(4 * hidden_dim, 256),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.slot_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, slot_dim),
            nn.LayerNorm(slot_dim),
        )
        self.topology_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, slot_dim),
            nn.LayerNorm(slot_dim),
        )
        self.static_charge = nn.Sequential(
            nn.Linear(static_charge_feature_dim, 2 * slot_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * slot_dim, slot_dim),
            nn.LayerNorm(slot_dim),
        )

    def encode_context(self, batch) -> dict[str, Tensor]:
        mask = batch.src_ids != self.pad_id
        hidden = self.token_emb(batch.src_ids) * math.sqrt(self.hidden_dim)
        hidden = hidden + self.segment_emb(batch.src_segment_ids)
        hidden = self.pos_emb(hidden)
        hidden = self.emb_drop(hidden)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)

        slot_positions = batch.slot_marker_positions[:, :7]
        topology_positions = batch.slot_marker_positions[:, 7]
        slot_states = hidden.gather(
            1,
            slot_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1)),
        )
        topology_state = hidden.gather(
            1,
            topology_positions.view(-1, 1, 1).expand(-1, 1, hidden.size(-1)),
        ).squeeze(1)
        slot_states = self.slot_proj(slot_states)
        topology_state = self.topology_proj(topology_state)
        static_charge = self.static_charge(batch.static_charge_features)
        return {
            "slot_states": slot_states,
            "channel_states": slot_states,
            "fermion_states": slot_states,
            "static_charge": static_charge,
            "topology_state": topology_state,
        }

    def get_stream_embeddings(self, batch) -> dict[str, Tensor]:
        encoded = self.encode_context(batch)
        return {
            "channel_prop": encoded["channel_states"],
            "fermion_line": encoded["fermion_states"],
            "fused": encoded["slot_states"],
        }
