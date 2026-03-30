"""Fixed-slot QED custom model with sequence decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .contracts import NUM_CHANNELS, NUM_PATTERNS, NUM_PROCESS_FAMILIES
from .encoder import CustomQEDFd2SqEncoder
from .factorization import (
    TARGET_VARIANT_FACTORIZED,
    normalize_target_variant,
)
from .features import EXTERNAL_MASS_SUMMARY_DIM, SLOT_FEATURE_DIM
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
                content_lengths += grammar.batch_is_content(tok_flat).long() * active.long()
                in_number = torch.where(active, next_in_number, in_number)
                number_has_digit = torch.where(active, next_number_has_digit, number_has_digit)
            finished = finished | (tok_flat == eos_id)
            if finished.all():
                break

        return tokens


@dataclass
class CustomQEDFd2SqModelOutput:
    sequence_logits: Tensor


@dataclass
class CustomQEDFd2SqGeneration:
    sequence_ids: Tensor


class CustomQEDFd2SqModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        node_in_dim: int = SLOT_FEATURE_DIM,
        hidden_dim: int = 128,
        stream_dim: int = 64,
        slot_dim: int | None = None,
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
        **_: int,
    ):
        super().__init__()
        self.encoder_variant = encoder_variant
        self.target_variant = normalize_target_variant(target_variant)
        self.use_grammar = use_grammar
        model_dim = slot_dim or stream_dim
        self.encoder = self._build_encoder(
            encoder_variant=encoder_variant,
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            slot_dim=model_dim,
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
            encoder_dim=model_dim,
            nhead=dec_nhead,
            num_layers=dec_layers,
            dim_ff=dec_dim_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_id=pad_id,
        )

        self.channel_emb = nn.Embedding(NUM_CHANNELS, model_dim)
        self.family_emb = nn.Embedding(NUM_PROCESS_FAMILIES, model_dim)
        self.pattern_emb = nn.Embedding(NUM_PATTERNS, model_dim)
        self.mass_proj = nn.Sequential(
            nn.Linear(EXTERNAL_MASS_SUMMARY_DIM, 2 * model_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.topology_token = nn.Sequential(
            nn.Linear(4 * model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.full_prefactor_context = nn.Sequential(
            nn.Linear(9 * model_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, model_dim),
        )
        self.full_denominator_context = nn.Sequential(
            nn.Linear(5 * model_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, model_dim),
        )
        self.channel_stream_context = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.fermion_stream_context = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.static_charge_context = nn.Sequential(
            nn.Linear(3 * model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.summary_token = nn.Sequential(
            nn.Linear(4 * model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

    @staticmethod
    def _build_encoder(
        encoder_variant: str,
        node_in_dim: int,
        hidden_dim: int,
        slot_dim: int,
        num_mp_layers: int,
        num_heads: int,
        dropout: float,
        src_vocab_size: int | None,
        src_pad_id: int | None,
        max_src_seq_len: int,
    ) -> nn.Module:
        if encoder_variant == "custom":
            return CustomQEDFd2SqEncoder(
                node_in_dim=node_in_dim,
                hidden_dim=hidden_dim,
                slot_dim=slot_dim,
                num_mp_layers=num_mp_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        if encoder_variant == "seq2seq":
            if src_vocab_size is None or src_pad_id is None:
                raise ValueError("Sequence encoder requires src_vocab_size and src_pad_id.")
            return SequenceDiagramEncoder(
                vocab_size=src_vocab_size,
                pad_id=src_pad_id,
                hidden_dim=hidden_dim,
                slot_dim=slot_dim,
                num_layers=num_mp_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=max_src_seq_len,
            )
        raise ValueError(
            f"Unknown encoder_variant '{encoder_variant}'. "
            "Expected 'custom' or 'seq2seq'."
        )

    def _head_inputs(self, batch, encoded: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        slot_states = encoded["slot_states"]
        ext1, ext2, ext3, ext4 = slot_states[:, 0], slot_states[:, 1], slot_states[:, 2], slot_states[:, 3]
        prop = slot_states[:, 6]
        static_charge = encoded["static_charge"]
        channel = self.channel_emb(batch.channel_id)
        family = self.family_emb(batch.process_family_id)
        pattern = self.pattern_emb(batch.pattern_id)
        mass_summary = self.mass_proj(batch.external_mass_summary)

        prefactor_input = torch.cat(
            [ext1, ext2, ext3, ext4, prop, static_charge, channel, family, pattern],
            dim=-1,
        )
        denominator_input = torch.cat(
            [prop, channel, family, mass_summary, pattern],
            dim=-1,
        )
        return prefactor_input, denominator_input

    def _full_sequence_memory(
        self,
        batch,
        encoded: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        prefactor_input, denominator_input = self._head_inputs(batch, encoded)
        channel_embed = self.channel_emb(batch.channel_id)
        family_embed = self.family_emb(batch.process_family_id)
        pattern_embed = self.pattern_emb(batch.pattern_id)
        channel_context = self.channel_stream_context(
            torch.cat(
                [
                    encoded["channel_states"].mean(dim=1),
                    channel_embed,
                ],
                dim=-1,
            )
        ).unsqueeze(1)
        fermion_context = self.fermion_stream_context(
            torch.cat(
                [
                    encoded["fermion_states"].mean(dim=1),
                    family_embed,
                ],
                dim=-1,
            )
        ).unsqueeze(1)
        static_charge_context = self.static_charge_context(
            torch.cat(
                [
                    encoded["static_charge"],
                    pattern_embed,
                    channel_embed,
                ],
                dim=-1,
            )
        ).unsqueeze(1)
        topology_context = self.summary_token(
            torch.cat(
                [
                    encoded["topology_state"],
                    channel_embed,
                    family_embed,
                    pattern_embed,
                ],
                dim=-1,
            )
        ).unsqueeze(1)
        memory = torch.cat(
            [
                encoded["slot_states"][:, 0:4, :],
                encoded["slot_states"][:, 6:7, :],
                encoded["slot_states"][:, 4:6, :],
                channel_context,
                fermion_context,
                static_charge_context,
                topology_context,
                self.full_prefactor_context(prefactor_input).unsqueeze(1),
                self.full_denominator_context(denominator_input).unsqueeze(1),
            ],
            dim=1,
        )
        mask = torch.ones(memory.size(0), memory.size(1), dtype=torch.bool, device=memory.device)
        return memory, mask

    def forward(
        self,
        batch,
        tgt_sequence: Tensor,
    ) -> CustomQEDFd2SqModelOutput:
        encoded = self.encoder.encode_context(batch)
        memory, memory_mask = self._full_sequence_memory(
            batch,
            encoded,
        )
        sequence_logits = self.decoder(
            tgt_sequence,
            memory,
            memory_pad_mask=~memory_mask,
        )
        return CustomQEDFd2SqModelOutput(sequence_logits=sequence_logits)

    @torch.no_grad()
    def generate(
        self,
        batch,
        max_len: int,
        sos_id: int,
        eos_id: int,
        grammar: "RPNGrammar | None" = None,
    ) -> CustomQEDFd2SqGeneration:
        if self.use_grammar and grammar is None:
            raise ValueError("`grammar` is required when use_grammar=True.")

        encoded = self.encoder.encode_context(batch)
        memory, memory_mask = self._full_sequence_memory(
            batch,
            encoded,
        )
        sequence_ids = self.decoder.generate(
            memory,
            max_len=max_len,
            sos_id=sos_id,
            eos_id=eos_id,
            memory_pad_mask=~memory_mask,
            grammar=grammar if self.use_grammar else None,
        )
        return CustomQEDFd2SqGeneration(sequence_ids=sequence_ids)
