"""Typed source-token serialization for canonical QED amplitudes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor


PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"
SPECIAL_TOKENS: tuple[str, ...] = (PAD, SOS, EOS, UNK)

SRC_TYPE_PAD = 0
SRC_TYPE_RAW = 1
SRC_TYPE_GLOBAL = 2
SRC_TYPE_FAMILY = 3
SRC_TYPE_TERM = 4
NUM_SOURCE_TYPES = 5

_RAW_SOURCE_TOKEN_RE = re.compile(
    r"\d+/\d+"
    r"|\d+"
    r"|[A-Za-z%\\_][A-Za-z0-9%\\_]*"
    r"|[{}(),*+\-/^]"
)
_RAW_INDEX_TOKEN_RE = re.compile(r"(.+)_([0-9]+)$")
_RAW_MOMENTUM_TOKEN_RE = re.compile(r"p_\d+$")
_RAW_MANDELSTAM_TOKEN_RE = re.compile(r"s_\d+$")
_RAW_MASS_TOKEN_RE = re.compile(r"m_[A-Za-z]+$")


class CanonicalSourceTokenizer:
    def __init__(self, token2id: dict[str, int] | None = None):
        if token2id is not None:
            self.token2id = dict(token2id)
        else:
            self.token2id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self._rebuild_reverse()

    def _rebuild_reverse(self) -> None:
        self.id2token = {value: key for key, value in self.token2id.items()}
        self.vocab_size = len(self.token2id)
        self.pad_id = self.token2id[PAD]
        self.sos_id = self.token2id[SOS]
        self.eos_id = self.token2id[EOS]
        self.unk_id = self.token2id[UNK]

    def build_vocab(self, sequences: Sequence[Sequence[str]]) -> "CanonicalSourceTokenizer":
        self.token2id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        for sequence in sequences:
            for token in sequence:
                if token not in self.token2id:
                    self.token2id[token] = len(self.token2id)
        self._rebuild_reverse()
        return self

    def encode_tokens(
        self,
        tokens: Sequence[str],
        max_len: int,
        add_sos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        ids: list[int] = []
        if add_sos:
            ids.append(self.sos_id)
        ids.extend(self.token2id.get(token, self.unk_id) for token in tokens)
        if add_eos:
            ids.append(self.eos_id)
        if len(ids) > max_len:
            ids = ids[: max_len - 1] + [self.eos_id]
        return ids

    def encode_tensor(
        self,
        tokens: Sequence[str],
        max_len: int,
        add_sos: bool = True,
        add_eos: bool = True,
    ) -> Tensor:
        ids = self.encode_tokens(tokens, max_len=max_len, add_sos=add_sos, add_eos=add_eos)
        out = torch.full((max_len,), self.pad_id, dtype=torch.long)
        out[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump({"token2id": self.token2id}, handle, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CanonicalSourceTokenizer":
        with open(path) as handle:
            payload = json.load(handle)
        return cls(token2id=payload["token2id"])


def flatten_term_sequences(term_sequences: Iterable[Sequence[str]]) -> list[str]:
    flattened: list[str] = ["[AMP]"]
    for sequence in term_sequences:
        flattened.extend(sequence)
    flattened.append("[END_AMP]")
    return flattened


def _normalize_raw_source_tokens(tokens: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    index_map: dict[str, str] = {}
    momentum_map: dict[str, str] = {}

    for token in tokens:
        if _RAW_MOMENTUM_TOKEN_RE.fullmatch(token):
            mapped = momentum_map.setdefault(token, f"MOMENTUM_{len(momentum_map)}")
            normalized.append(mapped)
            continue

        if (
            token in {"e", "i", "reg_prop"}
            or _RAW_MANDELSTAM_TOKEN_RE.fullmatch(token)
            or _RAW_MASS_TOKEN_RE.fullmatch(token)
        ):
            normalized.append(token)
            continue

        match = _RAW_INDEX_TOKEN_RE.fullmatch(token)
        if match is not None:
            prefix, _ = match.groups()
            mapped = index_map.setdefault(token, f"{prefix}_INDEX_{len(index_map)}")
            normalized.append(mapped)
            continue

        normalized.append(token)

    return normalized


def tokenize_raw_amplitude(raw_amplitude: str) -> list[str]:
    expr = re.sub(r"\s+", "", raw_amplitude.strip())
    if not expr:
        return []
    return _normalize_raw_source_tokens(_RAW_SOURCE_TOKEN_RE.findall(expr))


def tokenize_family_signature(signature: str) -> list[str]:
    normalized = signature.replace("|", " | ").replace(":", " : ")
    return [token for token in normalized.split() if token]


def encode_source_type_tensor(
    type_ids: Sequence[int],
    max_len: int,
    add_sos: bool = True,
    add_eos: bool = True,
) -> Tensor:
    ids: list[int] = []
    if add_sos:
        ids.append(SRC_TYPE_PAD)
    ids.extend(type_ids)
    if add_eos:
        ids.append(SRC_TYPE_PAD)
    if len(ids) > max_len:
        ids = ids[: max_len - 1] + [SRC_TYPE_PAD]
    out = torch.full((max_len,), SRC_TYPE_PAD, dtype=torch.long)
    out[: len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out
