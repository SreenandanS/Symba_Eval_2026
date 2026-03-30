"""
Custom graph encoder with three symmetry-aware message-passing streams.

The encoder consumes the explicit graph contract emitted by
`feynman_graph.py`:

- static kinematic relations
- explicit color-flow relations
- directed spinor / fermion-line relations
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax, to_dense_batch

from .color_mp import ColorFlowBlock
from .lorentz_mp import StaticKinematicBlock
from .spinor_mp import SpinorFlowBlock


class CrossStreamAttention(nn.Module):
    """Fuse kinematic, color, and spinor node states."""

    def __init__(
        self,
        stream_dim: int,
        num_streams: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=stream_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.stream_gate = nn.Sequential(
            nn.Linear(num_streams * stream_dim, num_streams),
            nn.Softmax(dim=-1),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(num_streams * stream_dim, 2 * stream_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * stream_dim, stream_dim),
        )
        self.layer_norm = nn.LayerNorm(stream_dim)

    def forward(self, kinematic_h: Tensor, color_h: Tensor, spinor_h: Tensor) -> Tensor:
        stacked = torch.stack([kinematic_h, color_h, spinor_h], dim=1)
        attn_out, _ = self.cross_attn(stacked, stacked, stacked)
        attn_out = stacked + attn_out

        flat = attn_out.reshape(attn_out.size(0), -1)
        gates = self.stream_gate(flat).unsqueeze(-1)
        gated = (attn_out * gates).reshape(attn_out.size(0), -1)
        fused = self.fusion_mlp(gated)
        return self.layer_norm(fused)


class CrossStreamExchange(nn.Module):
    """Layerwise non-local exchange between the three symmetry streams."""

    def __init__(
        self,
        stream_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=stream_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(stream_dim)
        self.ffn = nn.Sequential(
            nn.Linear(stream_dim, 2 * stream_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * stream_dim, stream_dim),
        )
        self.ffn_norm = nn.LayerNorm(stream_dim)

    def forward(
        self,
        kinematic_h: Tensor,
        color_h: Tensor,
        spinor_h: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        stacked = torch.stack([kinematic_h, color_h, spinor_h], dim=1)
        attn_out, _ = self.cross_attn(stacked, stacked, stacked)
        stacked = self.attn_norm(stacked + attn_out)
        stacked = self.ffn_norm(stacked + self.ffn(stacked))
        return stacked[:, 0], stacked[:, 1], stacked[:, 2]


class GraphReadout(nn.Module):
    """Graph-level pooling over fused node states."""

    def __init__(self, node_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn_weight = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.Tanh(),
            nn.Linear(node_dim // 2, 1),
        )
        self.external_transform = nn.Linear(node_dim, node_dim)
        self.vertex_transform = nn.Linear(node_dim, node_dim)
        self.propagator_transform = nn.Linear(node_dim, node_dim)
        self.readout_mlp = nn.Sequential(
            nn.Linear(3 * node_dim, 2 * out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * out_dim, out_dim),
        )

    def forward(
        self,
        x: Tensor,
        batch: Tensor,
        node_type_mask: Tensor | None = None,
    ) -> Tensor:
        mean_pool = global_mean_pool(x, batch)

        attn_scores = self.attn_weight(x)
        attn_scores = softmax(attn_scores, batch)
        attn_pool = global_add_pool(x * attn_scores, batch)

        if node_type_mask is not None:
            ext_mask = node_type_mask[:, 0:1]
            vert_mask = node_type_mask[:, 1:2]
            prop_mask = node_type_mask[:, 2:3]
            typed_x = (
                ext_mask * self.external_transform(x)
                + vert_mask * self.vertex_transform(x)
                + prop_mask * self.propagator_transform(x)
            )
            phys_pool = global_add_pool(typed_x, batch)
        else:
            phys_pool = global_max_pool(x, batch)

        combined = torch.cat([mean_pool, attn_pool, phys_pool], dim=-1)
        return self.readout_mlp(combined)


class CustomQCDFd2SqEncoder(nn.Module):
    """Three-stream QCD encoder built on the new explicit graph contract."""

    def __init__(
        self,
        node_in_dim: int = 19,
        edge_in_dim: int = 15,
        hidden_dim: int = 128,
        stream_dim: int = 64,
        graph_dim: int = 128,
        num_mp_layers: int = 3,
        num_heads: int = 4,
        mandelstam_dim: int = 6,
        color_feature_dim: int = 8,
        fermion_feature_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        del edge_in_dim, mandelstam_dim, color_feature_dim, fermion_feature_dim

        self.node_in_dim = node_in_dim
        self.graph_dim = graph_dim
        self.num_mp_layers = num_mp_layers

        self.kinematic_block = StaticKinematicBlock(
            in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            out_dim=stream_dim,
            num_layers=num_mp_layers,
            dropout=dropout,
        )
        # Alias used by stream inspection helpers.
        self.lorentz_block = self.kinematic_block

        self.color_block = ColorFlowBlock(
            in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            out_dim=stream_dim,
            num_layers=num_mp_layers,
            dropout=dropout,
        )
        self.spinor_block = SpinorFlowBlock(
            in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            out_dim=stream_dim,
            num_layers=num_mp_layers,
            dropout=dropout,
        )
        self.layer_exchanges = nn.ModuleList(
            CrossStreamExchange(
                stream_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_mp_layers)
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

    @staticmethod
    def _require_attr(data: Data | Batch, name: str) -> Tensor:
        value = getattr(data, name, None)
        if value is None:
            raise AttributeError(
                f"Graph is missing required custom-encoder attribute `{name}`. "
                "Build the dataset with the required graph contract."
            )
        return value

    def _run_streams(
        self,
        data: Data | Batch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        node_features = data.x
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(
                node_features.size(0),
                dtype=torch.long,
                device=node_features.device,
            )
        )

        node_role = self._require_attr(data, "node_role")
        kin_edge_index = self._require_attr(data, "kin_edge_index")
        edge_kin_relation = self._require_attr(data, "edge_kin_relation")
        node_momentum_signature = self._require_attr(data, "node_momentum_signature")
        node_mass_features = self._require_attr(data, "node_mass_features")
        kin_edge_channel = self._require_attr(data, "kin_edge_channel")

        color_edge_index = self._require_attr(data, "color_edge_index")
        edge_color_relation = self._require_attr(data, "edge_color_relation")
        node_color_rep = self._require_attr(data, "node_color_rep")

        spinor_edge_index = self._require_attr(data, "spinor_edge_index")
        edge_spinor_relation = self._require_attr(data, "edge_spinor_relation")
        fermion_line_id = self._require_attr(data, "fermion_line_id")
        node_fermion_num = self._require_attr(data, "node_fermion_num")
        vertex_interaction_type = self._require_attr(data, "vertex_interaction_type")

        kinematic_h = self.kinematic_block.encode_input(
            scalars=node_features,
            node_momentum_signature=node_momentum_signature,
            node_role=node_role,
            node_mass_features=node_mass_features,
        )
        color_h = self.color_block.encode_input(
            scalars=node_features,
            node_color_rep=node_color_rep,
            node_role=node_role,
        )
        spinor_h = self.spinor_block.encode_input(
            scalars=node_features,
            node_fermion_num=node_fermion_num,
            node_role=node_role,
            vertex_interaction_type=vertex_interaction_type,
        )
        for layer_idx in range(self.num_mp_layers):
            kinematic_h = self.kinematic_block.step_layer(
                layer_idx=layer_idx,
                h=kinematic_h,
                edge_index=kin_edge_index,
                edge_relation=edge_kin_relation,
                node_momentum_signature=node_momentum_signature,
                node_role=node_role,
                node_mass_features=node_mass_features,
                edge_channel=kin_edge_channel,
            )
            color_h = self.color_block.step_layer(
                layer_idx=layer_idx,
                h=color_h,
                edge_index=color_edge_index,
                edge_relation=edge_color_relation,
                node_color_rep=node_color_rep,
                node_role=node_role,
            )
            spinor_h = self.spinor_block.step_layer(
                layer_idx=layer_idx,
                h=spinor_h,
                edge_index=spinor_edge_index,
                edge_relation=edge_spinor_relation,
                fermion_line_id=fermion_line_id,
                node_fermion_num=node_fermion_num,
                node_role=node_role,
                vertex_interaction_type=vertex_interaction_type,
            )
            kinematic_h, color_h, spinor_h = self.layer_exchanges[layer_idx](
                kinematic_h,
                color_h,
                spinor_h,
            )
        kinematic_h = self.kinematic_block.project_output(kinematic_h)
        color_h = self.color_block.project_output(color_h)
        spinor_h = self.spinor_block.project_output(spinor_h)
        fused = self.fusion(kinematic_h, color_h, spinor_h)
        return fused, batch, kinematic_h, color_h, spinor_h

    def forward(self, data: Data | Batch) -> Tensor:
        fused, batch, *_ = self._run_streams(data)
        node_type_mask = data.x[:, -3:]
        return self.readout(fused, batch, node_type_mask)

    def encode_nodes(self, data: Data | Batch) -> tuple[Tensor, Tensor]:
        fused, batch, *_ = self._run_streams(data)
        memory, mask = to_dense_batch(fused, batch)
        return memory, mask

    def encode_context(self, data: Data | Batch) -> dict[str, Tensor]:
        fused, batch, kinematic_h, color_h, spinor_h = self._run_streams(data)
        memory, mask = to_dense_batch(fused, batch)
        return {
            "memory": memory,
            "memory_mask": mask,
            "batch": batch,
            "kinematic": kinematic_h,
            "color": color_h,
            "spinor": spinor_h,
            "fused": fused,
        }

    def get_stream_embeddings(self, data: Data | Batch) -> dict[str, Tensor]:
        fused, _batch, kinematic_h, color_h, spinor_h = self._run_streams(data)
        return {
            "kinematic": kinematic_h,
            "lorentz": kinematic_h,
            "color": color_h,
            "spinor": spinor_h,
            "fused": fused,
        }
