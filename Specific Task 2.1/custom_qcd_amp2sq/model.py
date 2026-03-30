"""Models for physics-informed amplitude -> squared-amplitude decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .encoder import (
    FlatSequenceEncoder,
    CustomQCDAmp2SqEncoder,
)
from .factorization import (
    TARGET_VARIANT_FACTORIZED,
    normalize_target_variant,
)
from .grammar import RPNGrammar


def _ensure_finite(name: str, tensor: Tensor) -> Tensor:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite tensor encountered in {name}.")
    return tensor


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
        encoder_dim: int = 96,
        nhead: int = 4,
        num_layers: int = 4,
        dim_ff: int = 512,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = SinusoidalPosEmb(d_model, max_len=max_seq_len * 2)
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
        return _ensure_finite("SeqDecoder.forward", self.out_proj(out))

    @torch.no_grad()
    def generate(
        self,
        memory: Tensor,
        max_len: int,
        sos_id: int,
        eos_id: int,
        memory_pad_mask: Tensor | None = None,
        grammar: RPNGrammar | None = None,
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
                empty_valid = (~valid.any(dim=-1)) & ~finished
                if empty_valid.any():
                    raise RuntimeError(
                        "Grammar produced no valid next token for an active sample."
                    )
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
class Amp2SqModelOutput:
    sequence_logits: Tensor


@dataclass
class Amp2SqGeneration:
    sequence_ids: Tensor


class Amp2SqModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        node_in_dim: int,
        src_vocab_size: int,
        src_pad_id: int,
        hidden_dim: int = 128,
        stream_dim: int = 96,
        graph_dim: int = 128,
        num_mp_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        dec_d_model: int = 256,
        dec_nhead: int = 4,
        dec_layers: int = 4,
        dec_dim_ff: int = 512,
        max_tgt_seq_len: int = 256,
        max_src_term_len: int = 192,
        max_flat_src_len: int = 3072,
        pad_id: int = 0,
        encoder_variant: str = "custom",
        target_variant: str = TARGET_VARIANT_FACTORIZED,
        use_grammar: bool = True,
    ):
        super().__init__()
        self.encoder_variant = encoder_variant
        self.target_variant = normalize_target_variant(target_variant)
        self.use_grammar = use_grammar
        self.stream_dim = stream_dim

        self.encoder = self._build_encoder(
            encoder_variant=encoder_variant,
            node_in_dim=node_in_dim,
            src_vocab_size=src_vocab_size,
            src_pad_id=src_pad_id,
            hidden_dim=hidden_dim,
            stream_dim=stream_dim,
            graph_dim=graph_dim,
            num_mp_layers=num_mp_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_src_term_len=max_src_term_len,
            max_flat_src_len=max_flat_src_len,
        )
        self.decoder = SeqDecoder(
            vocab_size=vocab_size,
            d_model=dec_d_model,
            encoder_dim=stream_dim,
            nhead=dec_nhead,
            num_layers=dec_layers,
            dim_ff=dec_dim_ff,
            max_seq_len=max_tgt_seq_len,
            dropout=dropout,
            pad_id=pad_id,
        )

        self.full_raw_context = nn.Sequential(
            nn.Linear(graph_dim + 2 * stream_dim, stream_dim),
            nn.SiLU(),
            nn.Linear(stream_dim, stream_dim),
        )
        self.full_feature_context = nn.Sequential(
            nn.Linear(graph_dim + 2 * stream_dim, stream_dim),
            nn.SiLU(),
            nn.Linear(stream_dim, stream_dim),
        )
        self.full_family_context = nn.Sequential(
            nn.Linear(graph_dim + 2 * stream_dim, stream_dim),
            nn.SiLU(),
            nn.Linear(stream_dim, stream_dim),
        )
        self.global_condition = nn.Sequential(
            nn.Linear(graph_dim + 2 * stream_dim, stream_dim),
            nn.SiLU(),
            nn.Linear(stream_dim, stream_dim),
        )

    @staticmethod
    def _build_encoder(
        *,
        encoder_variant: str,
        node_in_dim: int,
        src_vocab_size: int,
        src_pad_id: int,
        hidden_dim: int,
        stream_dim: int,
        graph_dim: int,
        num_mp_layers: int,
        num_heads: int,
        dropout: float,
        max_src_term_len: int,
        max_flat_src_len: int,
    ) -> nn.Module:
        if encoder_variant == "custom":
            return CustomQCDAmp2SqEncoder(
                node_in_dim=node_in_dim,
                src_vocab_size=src_vocab_size,
                src_pad_id=src_pad_id,
                hidden_dim=hidden_dim,
                stream_dim=stream_dim,
                graph_dim=graph_dim,
                num_mp_layers=num_mp_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_src_term_len=max_src_term_len,
                max_flat_src_len=max_flat_src_len,
            )
        if encoder_variant == "seq2seq":
            return FlatSequenceEncoder(
                src_vocab_size=src_vocab_size,
                src_pad_id=src_pad_id,
                hidden_dim=hidden_dim,
                stream_dim=stream_dim,
                graph_dim=graph_dim,
                num_layers=num_mp_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_src_seq_len=max_flat_src_len,
            )
        raise ValueError(
            f"Unknown encoder_variant '{encoder_variant}'. "
            "Expected 'custom' or 'seq2seq'."
        )

    def _pooled_context(self, encoded: dict[str, Tensor]) -> dict[str, Tensor]:
        pooled = encoded["pooled"]
        raw_pool = encoded["raw_pool"]
        feature_pool = encoded["feature_pool"]
        family_pool = encoded["family_pool"]
        global_pool = encoded["global_pool"]
        return {
            "pooled": pooled,
            "raw_pool": raw_pool,
            "feature_pool": feature_pool,
            "family_pool": family_pool,
            "global_pool": global_pool,
        }

    def _full_sequence_memory(
        self,
        encoded: dict[str, Tensor],
        pooled: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        raw_token = self.full_raw_context(
            torch.cat(
                [
                    pooled["pooled"],
                    pooled["raw_pool"],
                    pooled["global_pool"],
                ],
                dim=-1,
            )
        ).unsqueeze(1)
        feature_token = self.full_feature_context(
            torch.cat(
                [
                    pooled["pooled"],
                    pooled["feature_pool"],
                    pooled["global_pool"],
                ],
                dim=-1,
            )
        ).unsqueeze(1)
        family_token = self.full_family_context(
            torch.cat(
                [
                    pooled["pooled"],
                    pooled["family_pool"],
                    pooled["feature_pool"],
                ],
                dim=-1,
            )
        ).unsqueeze(1)
        global_token = self.global_condition(
            torch.cat(
                [
                    pooled["pooled"],
                    pooled["feature_pool"],
                    pooled["global_pool"],
                ],
                dim=-1,
            )
        ).unsqueeze(1)
        memory = torch.cat(
            [raw_token, feature_token, family_token, global_token, encoded["memory"]],
            dim=1,
        )
        cond_mask = torch.ones(
            (encoded["memory_mask"].size(0), 4),
            dtype=torch.bool,
            device=encoded["memory_mask"].device,
        )
        memory_mask = torch.cat([cond_mask, encoded["memory_mask"]], dim=1)
        return memory, memory_mask

    def forward(
        self,
        batch,
        tgt_tokens: Tensor,
    ) -> Amp2SqModelOutput:
        encoded = self.encoder.encode_context(batch)
        pooled = self._pooled_context(encoded)
        memory, memory_mask = self._full_sequence_memory(encoded, pooled)
        sequence_logits = self.decoder(
            tgt_tokens,
            memory,
            memory_pad_mask=~memory_mask,
        )
        sequence_logits = _ensure_finite(
            "Amp2SqModel.decoder",
            sequence_logits,
        )
        return Amp2SqModelOutput(sequence_logits=sequence_logits)

    @torch.no_grad()
    def generate(
        self,
        batch,
        max_len: int,
        sos_id: int,
        eos_id: int,
        grammar: RPNGrammar | None = None,
    ) -> Amp2SqGeneration:
        if self.use_grammar and grammar is None:
            raise ValueError("`grammar` is required when use_grammar=True.")

        encoded = self.encoder.encode_context(batch)
        pooled = self._pooled_context(encoded)
        memory, memory_mask = self._full_sequence_memory(encoded, pooled)
        sequence_ids = self.decoder.generate(
            memory=memory,
            max_len=max_len,
            sos_id=sos_id,
            eos_id=eos_id,
            memory_pad_mask=~memory_mask,
            grammar=grammar if self.use_grammar else None,
        )
        return Amp2SqGeneration(sequence_ids=sequence_ids)
