"""Sequence encoders for QED amplitude -> squared-amplitude prediction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .sequence_utils import (
    NUM_SOURCE_TYPES,
    SRC_TYPE_FAMILY,
    SRC_TYPE_GLOBAL,
    SRC_TYPE_RAW,
    SRC_TYPE_TERM,
)


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    if x.dim() != 3:
        raise ValueError("Expected [batch, seq, dim] tensor.")
    weights = mask.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_mean_where(x: Tensor, base_mask: Tensor, type_ids: Tensor, value: int) -> Tensor:
    return _masked_mean(x, base_mask & (type_ids == value))


def _ensure_finite(name: str, tensor: Tensor) -> Tensor:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite tensor encountered in {name}.")
    return tensor


class SinusoidalPosEmb(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


class SourceSequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        hidden_dim: int = 128,
        out_dim: int = 96,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        num_type_embeddings: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim
        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.type_emb = (
            nn.Embedding(num_type_embeddings, hidden_dim)
            if num_type_embeddings > 0
            else None
        )
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
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _encode_tokens(
        self,
        token_ids: Tensor,
        type_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mask = token_ids != self.pad_id
        valid_rows = mask.any(dim=1)
        encoded = torch.zeros(
            token_ids.size(0),
            token_ids.size(1),
            self.out_proj[-1].out_features,
            device=token_ids.device,
            dtype=self.token_emb.weight.dtype,
        )
        if valid_rows.any():
            token_ids_valid = token_ids[valid_rows]
            mask_valid = mask[valid_rows]
            x = self.token_emb(token_ids_valid) * math.sqrt(self.hidden_dim)
            if self.type_emb is not None and type_ids is not None:
                x = x + self.type_emb(type_ids[valid_rows])
            x = self.pos_emb(x)
            x = self.emb_drop(x)
            x = self.encoder(x, src_key_padding_mask=~mask_valid)
            x = self.out_proj(x)
            x = x * mask_valid.unsqueeze(-1).to(x.dtype)
            encoded[valid_rows] = x
        encoded = _ensure_finite("SourceSequenceEncoder", encoded)
        pooled = _masked_mean(encoded, mask)
        return encoded, pooled, mask

    def encode_sequence(
        self,
        src_ids: Tensor,
        src_type_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self._encode_tokens(src_ids, src_type_ids)


class CustomQEDAmp2SqEncoder(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 0,
        src_vocab_size: int = 0,
        src_pad_id: int = 0,
        hidden_dim: int = 128,
        stream_dim: int = 96,
        graph_dim: int = 128,
        num_mp_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_src_term_len: int = 192,
        max_flat_src_len: int = 3072,
    ):
        super().__init__()
        del node_in_dim, max_src_term_len
        source_layers = max(num_mp_layers + 2, 5)
        self.source_encoder = SourceSequenceEncoder(
            vocab_size=src_vocab_size,
            pad_id=src_pad_id,
            hidden_dim=hidden_dim,
            out_dim=stream_dim,
            num_layers=source_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_flat_src_len,
            num_type_embeddings=NUM_SOURCE_TYPES,
        )
        self.pooled_proj = nn.Sequential(
            nn.Linear(stream_dim * 4, graph_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_dim, graph_dim),
        )

    def encode_context(self, batch) -> dict[str, Tensor]:
        seq_memory, seq_pool, seq_mask = self.source_encoder.encode_sequence(
            batch.flat_src_ids,
            batch.flat_src_type_ids,
        )
        raw_pool = _masked_mean_where(
            seq_memory,
            seq_mask,
            batch.flat_src_type_ids,
            SRC_TYPE_RAW,
        )
        global_pool = _masked_mean_where(
            seq_memory,
            seq_mask,
            batch.flat_src_type_ids,
            SRC_TYPE_GLOBAL,
        )
        family_pool = _masked_mean_where(
            seq_memory,
            seq_mask,
            batch.flat_src_type_ids,
            SRC_TYPE_FAMILY,
        )
        term_pool = _masked_mean_where(
            seq_memory,
            seq_mask,
            batch.flat_src_type_ids,
            SRC_TYPE_TERM,
        )
        pooled = self.pooled_proj(
            torch.cat([raw_pool, global_pool, family_pool, term_pool], dim=-1)
        )
        return {
            "memory": seq_memory,
            "memory_mask": seq_mask,
            "pooled": pooled,
            "raw_pool": raw_pool,
            "feature_pool": term_pool,
            "family_pool": family_pool,
            "global_pool": global_pool,
        }


class FlatSequenceEncoder(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        src_pad_id: int,
        hidden_dim: int = 128,
        stream_dim: int = 96,
        graph_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_src_seq_len: int = 3072,
    ):
        super().__init__()
        self.source_encoder = SourceSequenceEncoder(
            vocab_size=src_vocab_size,
            pad_id=src_pad_id,
            hidden_dim=hidden_dim,
            out_dim=stream_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_src_seq_len,
        )
        self.pool_proj = nn.Sequential(
            nn.Linear(stream_dim, graph_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_dim, graph_dim),
        )

    def encode_context(self, batch) -> dict[str, Tensor]:
        seq_memory, seq_pool, seq_mask = self.source_encoder.encode_sequence(
            batch.flat_src_ids,
            batch.flat_src_type_ids,
        )
        pooled = self.pool_proj(seq_pool)
        return {
            "memory": seq_memory,
            "memory_mask": seq_mask,
            "pooled": pooled,
            "raw_pool": seq_pool,
            "feature_pool": seq_pool,
            "family_pool": seq_pool,
            "global_pool": seq_pool,
        }
