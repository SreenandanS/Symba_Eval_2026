"""Fixed-slot custom encoder for the QED tree-level 2->2 workflow."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .contracts import NUM_FIXED_SLOTS, NUM_NODE_ROLES
from .features import STATIC_CHARGE_FEATURE_DIM, TOPOLOGY_FEATURE_DIM


def _ensure_finite(name: str, tensor: Tensor) -> Tensor:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite tensor encountered in {name}.")
    return tensor


def _gather_nodes(h: Tensor, node_index: Tensor) -> Tensor:
    return h.gather(
        1,
        node_index.unsqueeze(-1).expand(-1, -1, h.size(-1)),
    )


class RelationStream(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        edge_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.msg_mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(3 * hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.ffns = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        )
        self.ffn_norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        )
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    @staticmethod
    def _aggregate(dst_index: Tensor, msg: Tensor, num_slots: int) -> Tensor:
        batch_size, edge_count, hidden_dim = msg.shape
        agg = msg.new_zeros(batch_size, num_slots, hidden_dim)
        agg.scatter_add_(
            1,
            dst_index.unsqueeze(-1).expand(-1, -1, hidden_dim),
            msg,
        )
        degree = msg.new_zeros(batch_size, num_slots, 1)
        degree.scatter_add_(
            1,
            dst_index.unsqueeze(-1),
            torch.ones(
                batch_size,
                edge_count,
                1,
                device=msg.device,
                dtype=msg.dtype,
            ),
        )
        return agg / degree.clamp_min(1.0)

    def step_layer(
        self,
        layer_idx: int,
        h: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        src_index = edge_index[..., 0]
        dst_index = edge_index[..., 1]
        src_h = _gather_nodes(h, src_index)
        dst_h = _gather_nodes(h, dst_index)
        rel_h = self.edge_proj(edge_attr)
        msg = self.msg_mlps[layer_idx](torch.cat([src_h, dst_h, rel_h], dim=-1))
        agg = self._aggregate(dst_index, msg, h.size(1))

        norm = self.norms[layer_idx]
        ffn = self.ffns[layer_idx]
        ffn_norm = self.ffn_norms[layer_idx]
        h = norm(h + agg)
        h = ffn_norm(h + ffn(h))
        return h

    def project_output(self, h: Tensor) -> Tensor:
        return _ensure_finite("QEDRelationStream", self.out_proj(h))


class CrossStreamFusion(nn.Module):
    def __init__(self, stream_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=stream_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(stream_dim * 3, 3),
            nn.Softmax(dim=-1),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(stream_dim * 3, 2 * stream_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * stream_dim, stream_dim),
        )
        self.norm = nn.LayerNorm(stream_dim)

    def forward(
        self,
        slot_h: Tensor,
        channel_h: Tensor,
        fermion_h: Tensor,
    ) -> Tensor:
        batch_size, num_slots, hidden_dim = slot_h.shape
        stacked = torch.stack([slot_h, channel_h, fermion_h], dim=2)
        stacked = stacked.view(batch_size * num_slots, 3, hidden_dim)
        attn_out, _ = self.cross_attn(stacked, stacked, stacked)
        stacked = stacked + attn_out
        flat = stacked.reshape(batch_size * num_slots, -1)
        gates = self.gate(flat).unsqueeze(-1)
        fused = self.fusion_mlp((stacked * gates).reshape(batch_size * num_slots, -1))
        fused = self.norm(fused)
        return fused.view(batch_size, num_slots, hidden_dim)


class StaticChargeFeatures(nn.Module):
    def __init__(
        self,
        in_dim: int = STATIC_CHARGE_FEATURE_DIM,
        out_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2 * out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, static_charge_features: Tensor) -> Tensor:
        return self.net(static_charge_features)


class CustomQEDFd2SqEncoder(nn.Module):
    """Fixed-slot custom encoder for the canonical 7-slot QED interaction contract."""

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 128,
        slot_dim: int = 64,
        num_mp_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        topology_feature_dim: int = TOPOLOGY_FEATURE_DIM,
        static_charge_feature_dim: int = STATIC_CHARGE_FEATURE_DIM,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.slot_type_emb = nn.Embedding(NUM_NODE_ROLES, hidden_dim)
        self.slot_position_emb = nn.Embedding(NUM_FIXED_SLOTS, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.slot_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, slot_dim),
        )
        self.channel_prop_stream = RelationStream(
            hidden_dim=hidden_dim,
            out_dim=slot_dim,
            edge_dim=4,
            num_layers=num_mp_layers,
            dropout=dropout,
        )
        self.fermion_line_stream = RelationStream(
            hidden_dim=hidden_dim,
            out_dim=slot_dim,
            edge_dim=3,
            num_layers=num_mp_layers,
            dropout=dropout,
        )
        self.fusion = CrossStreamFusion(
            stream_dim=slot_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.topology_proj = nn.Sequential(
            nn.Linear(3 * slot_dim + topology_feature_dim, 2 * slot_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * slot_dim, slot_dim),
            nn.LayerNorm(slot_dim),
        )
        self.static_charge = StaticChargeFeatures(
            in_dim=static_charge_feature_dim,
            out_dim=slot_dim,
            dropout=dropout,
        )

    def encode_context(self, batch) -> dict[str, Tensor]:
        base = self.node_proj(batch.slot_features)
        base = base + self.slot_type_emb(batch.slot_type_ids)
        base = base + self.slot_position_emb(batch.slot_position_ids)
        base = self.input_norm(base)

        channel_hidden = base
        fermion_hidden = base
        for layer_idx in range(len(self.channel_prop_stream.msg_mlps)):
            channel_hidden = self.channel_prop_stream.step_layer(
                layer_idx,
                channel_hidden,
                batch.channel_prop_edge_index,
                batch.channel_prop_edge_attr,
            )
            fermion_hidden = self.fermion_line_stream.step_layer(
                layer_idx,
                fermion_hidden,
                batch.fermion_line_edge_index,
                batch.fermion_line_edge_attr,
            )

        slot_states = _ensure_finite("CustomQEDFd2SqEncoder.slot_stream", self.slot_ffn(base))
        channel_states = self.channel_prop_stream.project_output(channel_hidden)
        fermion_states = self.fermion_line_stream.project_output(fermion_hidden)
        fused = self.fusion(slot_states, channel_states, fermion_states)
        fused = _ensure_finite("CustomQEDFd2SqEncoder.fusion", fused)

        topology_state = self.topology_proj(
            torch.cat(
                [
                    fused[:, 4, :],
                    fused[:, 5, :],
                    fused[:, 6, :],
                    batch.topology_features,
                ],
                dim=-1,
            )
        )
        static_charge = self.static_charge(batch.static_charge_features)
        memory_mask = torch.ones(
            fused.size(0),
            fused.size(1),
            dtype=torch.bool,
            device=fused.device,
        )
        return {
            "slot_states": fused,
            "channel_states": channel_states,
            "fermion_states": fermion_states,
            "static_charge": static_charge,
            "topology_state": topology_state,
            "memory": fused,
            "memory_mask": memory_mask,
        }

    def get_stream_embeddings(self, batch) -> dict[str, Tensor]:
        encoded = self.encode_context(batch)
        return {
            "channel_prop": encoded["channel_states"],
            "fermion_line": encoded["fermion_states"],
            "fused": encoded["slot_states"],
        }
