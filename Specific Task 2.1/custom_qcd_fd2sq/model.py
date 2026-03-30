"""Sequence decoder built on the QCD custom graph encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool, global_mean_pool

from .contracts import NODE_ROLE_PROPAGATOR
from .encoder import CustomQCDFd2SqEncoder
from .factorization import (
    TARGET_VARIANT_FACTORIZED,
    normalize_target_variant,
)
from .grammar import RPNGrammar
from .sequence_encoder import SequenceDiagramEncoder


class SinusoidalPosEmb(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
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


class SeqDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        encoder_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_ff: int = 512,
        max_seq_len: int = 1700,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = SinusoidalPosEmb(d_model, max_seq_len * 2)
        self.emb_drop = nn.Dropout(dropout)
        self.memory_proj = nn.Linear(encoder_dim, d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones((length, length), device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_pad_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.tok_emb(tgt) * math.sqrt(self.d_model)
        x = self.pos_emb(x)
        x = self.emb_drop(x)
        mem = self.memory_proj(memory)

        out = self.transformer(
            tgt=x,
            memory=mem,
            tgt_mask=self._causal_mask(tgt.size(1), tgt.device),
            tgt_key_padding_mask=(tgt == self.pad_id),
            memory_key_padding_mask=memory_pad_mask,
        )
        return self.out_proj(out)

    @torch.no_grad()
    def generate(
        self,
        memory: Tensor,
        max_len: int,
        sos_id: int,
        eos_id: int,
        memory_pad_mask: Tensor | None = None,
        grammar: "RPNGrammar | None" = None,
    ) -> Tensor:
        batch_size = memory.size(0)
        device = memory.device
        tokens = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if grammar is not None:
            stack_depths = torch.zeros(batch_size, dtype=torch.long, device=device)
            content_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
            in_number = torch.zeros(batch_size, dtype=torch.bool, device=device)
            number_has_digit = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(tokens, memory, memory_pad_mask)
            next_logits = logits[:, -1].clone()

            if grammar is not None:
                valid = grammar.get_valid_mask(
                    stack_depths=stack_depths,
                    generated_content_lengths=content_lengths,
                    in_number=in_number,
                    number_has_digit=number_has_digit,
                ).to(device)
                if finished.any():
                    valid[finished] = False
                    valid[finished, grammar.pad_id] = True
                next_logits[~valid] = float("-inf")
            elif finished.any():
                next_logits[finished] = float("-inf")
                next_logits[finished, self.pad_id] = 0.0

            next_tok = next_logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)

            tok_flat = next_tok.squeeze(-1)
            active = ~finished
            if grammar is not None:
                delta, next_in_number, next_number_has_digit = grammar.batch_transition(
                    tok_flat,
                    in_number,
                    number_has_digit,
                )
                stack_depths += delta * active.long()
                content_lengths += (
                    grammar.batch_is_content(tok_flat).long() * active.long()
                )
                in_number = torch.where(active, next_in_number, in_number)
                number_has_digit = torch.where(
                    active,
                    next_number_has_digit,
                    number_has_digit,
                )
            finished = finished | (tok_flat == eos_id)
            if finished.all():
                break

        return tokens


@dataclass
class CustomQCDFd2SqModelOutput:
    sequence_logits: Tensor


@dataclass
class CustomQCDFd2SqGeneration:
    sequence_ids: Tensor


class CustomQCDFd2SqModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        node_in_dim: int = 19,
        edge_in_dim: int = 15,
        hidden_dim: int = 128,
        stream_dim: int = 64,
        graph_dim: int = 128,
        num_mp_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        dec_d_model: int = 256,
        dec_nhead: int = 4,
        dec_layers: int = 4,
        dec_dim_ff: int = 512,
        max_seq_len: int = 1700,
        pad_id: int = 0,
        encoder_variant: str = "custom",
        target_variant: str = TARGET_VARIANT_FACTORIZED,
        use_grammar: bool = True,
        src_vocab_size: int | None = None,
        src_pad_id: int | None = None,
        max_src_seq_len: int = 256,
    ):
        super().__init__()
        self.encoder_variant = encoder_variant
        self.target_variant = normalize_target_variant(target_variant)
        self.use_grammar = use_grammar
        self.encoder = self._build_encoder(
            encoder_variant=encoder_variant,
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            stream_dim=stream_dim,
            graph_dim=graph_dim,
            num_mp_layers=num_mp_layers,
            num_heads=num_heads,
            dropout=dropout,
            src_vocab_size=src_vocab_size,
            src_pad_id=src_pad_id,
            max_src_seq_len=max_src_seq_len,
        )
        self.decoder = SeqDecoder(
            vocab_size=vocab_size,
            d_model=dec_d_model,
            encoder_dim=stream_dim,
            nhead=dec_nhead,
            num_layers=dec_layers,
            dim_ff=dec_dim_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_id=pad_id,
        )
        self.full_color_context = nn.Sequential(
            nn.Linear(stream_dim * 3, stream_dim),
            nn.SiLU(),
            nn.Linear(stream_dim, stream_dim),
        )
        self.full_denominator_context = nn.Sequential(
            nn.Linear(stream_dim * 3, stream_dim),
            nn.SiLU(),
            nn.Linear(stream_dim, stream_dim),
        )
        self.full_spinor_context = nn.Sequential(
            nn.Linear(stream_dim * 2, stream_dim),
            nn.SiLU(),
            nn.Linear(stream_dim, stream_dim),
        )

    @staticmethod
    def _build_encoder(
        encoder_variant: str,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int,
        stream_dim: int,
        graph_dim: int,
        num_mp_layers: int,
        num_heads: int,
        dropout: float,
        src_vocab_size: int | None,
        src_pad_id: int | None,
        max_src_seq_len: int,
    ) -> nn.Module:
        if encoder_variant == "custom":
            return CustomQCDFd2SqEncoder(
                node_in_dim=node_in_dim,
                edge_in_dim=edge_in_dim,
                hidden_dim=hidden_dim,
                stream_dim=stream_dim,
                graph_dim=stream_dim,
                num_mp_layers=num_mp_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        if encoder_variant == "seq2seq":
            if src_vocab_size is None or src_pad_id is None:
                raise ValueError(
                    "Sequence encoder requires src_vocab_size and src_pad_id."
                )
            return SequenceDiagramEncoder(
                vocab_size=src_vocab_size,
                pad_id=src_pad_id,
                hidden_dim=hidden_dim,
                stream_dim=stream_dim,
                graph_dim=graph_dim,
                num_layers=num_mp_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=max_src_seq_len,
            )
        raise ValueError(
            f"Unknown encoder_variant '{encoder_variant}'. "
            "Expected 'custom' or 'seq2seq'."
        )

    def _pooled_context(
        self,
        data,
        encoded: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        batch = encoded["batch"]
        prop_mask = encoded.get("propagator_mask")
        if prop_mask is None:
            node_role = data.node_role
            prop_mask = (node_role == NODE_ROLE_PROPAGATOR)
        prop_mask = prop_mask.to(encoded["kinematic"].dtype)
        prop_mask = prop_mask.unsqueeze(-1)

        color_pool = global_mean_pool(encoded["color"], batch)
        kinematic_pool = global_mean_pool(encoded["kinematic"], batch)
        spinor_pool = global_mean_pool(encoded["spinor"], batch)
        fused_pool = global_mean_pool(encoded["fused"], batch)
        propagator_pool = global_add_pool(encoded["kinematic"] * prop_mask, batch)

        return {
            "color_pool": color_pool,
            "kinematic_pool": kinematic_pool,
            "spinor_pool": spinor_pool,
            "fused_pool": fused_pool,
            "propagator_pool": propagator_pool,
        }

    def _full_sequence_memory(
        self,
        encoded: dict[str, Tensor],
        pooled: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        color_token = self.full_color_context(
            torch.cat(
                [
                    pooled["color_pool"],
                    pooled["fused_pool"],
                    pooled["spinor_pool"],
                ],
                dim=-1,
            )
        ).unsqueeze(1)
        denominator_token = self.full_denominator_context(
            torch.cat(
                [
                    pooled["kinematic_pool"],
                    pooled["propagator_pool"],
                    pooled["fused_pool"],
                ],
                dim=-1,
            )
        ).unsqueeze(1)
        spinor_token = self.full_spinor_context(
            torch.cat(
                [
                    pooled["spinor_pool"],
                    pooled["fused_pool"],
                ],
                dim=-1,
            )
        ).unsqueeze(1)

        memory = torch.cat(
            [color_token, denominator_token, spinor_token, encoded["memory"]],
            dim=1,
        )
        cond_mask = torch.ones(
            (encoded["memory_mask"].size(0), 3),
            dtype=torch.bool,
            device=encoded["memory_mask"].device,
        )
        memory_mask = torch.cat([cond_mask, encoded["memory_mask"]], dim=1)
        return memory, memory_mask

    def forward(
        self,
        data: Data | Batch,
        tgt_sequence: Tensor,
    ) -> CustomQCDFd2SqModelOutput:
        encoded = self.encoder.encode_context(data)
        pooled = self._pooled_context(data, encoded)
        memory, memory_mask = self._full_sequence_memory(encoded, pooled)
        sequence_logits = self.decoder(
            tgt_sequence,
            memory,
            memory_pad_mask=~memory_mask,
        )
        return CustomQCDFd2SqModelOutput(sequence_logits=sequence_logits)

    @torch.no_grad()
    def generate(
        self,
        data: Data | Batch,
        max_len: int,
        sos_id: int,
        eos_id: int,
        grammar: "RPNGrammar | None" = None,
    ) -> CustomQCDFd2SqGeneration:
        if self.use_grammar and grammar is None:
            raise ValueError("`grammar` is required when use_grammar=True.")

        encoded = self.encoder.encode_context(data)
        pooled = self._pooled_context(data, encoded)
        memory, memory_mask = self._full_sequence_memory(encoded, pooled)
        sequence_ids = self.decoder.generate(
            memory=memory,
            max_len=max_len,
            sos_id=sos_id,
            eos_id=eos_id,
            memory_pad_mask=~memory_mask,
            grammar=grammar if self.use_grammar else None,
        )
        return CustomQCDFd2SqGeneration(sequence_ids=sequence_ids)

    def get_embeddings(self, data: Data | Batch) -> dict[str, Tensor]:
        return self.encoder.get_stream_embeddings(data)
