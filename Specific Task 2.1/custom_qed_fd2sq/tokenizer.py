"""Tokenizer utilities for QED custom decoder targets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor


PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"

SPECIAL_TOKENS: list[str] = [PAD, SOS, EOS, UNK]

NUM_START = "[NUM_START]"
NUM_END = "[NUM_END]"
DIGIT_TOKENS: tuple[str, ...] = tuple(f"[DIGIT_{digit}]" for digit in range(10))

ADD = "[ADD]"
SUB = "[SUB]"
MUL = "[MUL]"
DIV = "[DIV]"
POW = "[POW]"
NEG = "[NEG]"

BINARY_OPS: frozenset[str] = frozenset({ADD, SUB, MUL, DIV, POW})
UNARY_OPS: frozenset[str] = frozenset({NEG})
ALL_OPS: frozenset[str] = BINARY_OPS | UNARY_OPS
DEFAULT_SYMBOL_TOKENS: tuple[str, ...] = (
    "m_b",
    "m_c",
    "m_d",
    "m_e",
    "m_mu",
    "m_s",
    "m_t",
    "m_tt",
    "m_u",
    "s_12",
    "s_13",
    "s_14",
    "s_23",
    "s_24",
    "s_34",
    "reg_prop",
)
INFIX_CONTROL_TOKENS: tuple[str, ...] = ("(", ")", "+", "-", "*", "/", "^")

_TOKEN_RE = re.compile(
    r"\d+/\d+"
    r"|"
    r"[a-zA-Z_][a-zA-Z_0-9]*"
    r"|"
    r"\d+"
    r"|[+\-*/^()]"
)
_INFIX_TOKEN_RE = re.compile(
    r"\d+/\d+"
    r"|[a-zA-Z_][a-zA-Z_0-9]*"
    r"|\d+"
    r"|[+\-*/^()]"
)


@dataclass
class _Literal:
    value: str


@dataclass
class _Var:
    name: str


@dataclass
class _BinOp:
    op: str
    left: "_ASTNode"
    right: "_ASTNode"


@dataclass
class _UnaryNeg:
    expr: "_ASTNode"


_ASTNode = Union[_Literal, _Var, _BinOp, _UnaryNeg]


def _is_rational_literal(token: str) -> bool:
    if token.count("/") != 1:
        return False
    numerator, denominator = token.split("/", 1)
    return numerator.isdigit() and denominator.isdigit()


class _InfixParser:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self) -> str:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _expect(self, expected: str) -> None:
        if self.pos >= len(self.tokens):
            raise SyntaxError(f"Expected '{expected}' but reached end of expression")
        token = self._consume()
        if token != expected:
            raise SyntaxError(
                f"Expected '{expected}', got '{token}' at token position {self.pos - 1}"
            )

    def parse(self) -> _ASTNode:
        node = self._expr()
        if self.pos < len(self.tokens):
            raise SyntaxError(
                f"Unexpected trailing token '{self.tokens[self.pos]}' at position {self.pos}"
            )
        return node

    def _expr(self) -> _ASTNode:
        node = self._term()
        while self._peek() in ("+", "-"):
            op = self._consume()
            node = _BinOp(op, node, self._term())
        return node

    def _term(self) -> _ASTNode:
        node = self._power()
        while self._peek() in ("*", "/"):
            op = self._consume()
            node = _BinOp(op, node, self._power())
        return node

    def _power(self) -> _ASTNode:
        node = self._unary()
        if self._peek() == "^":
            self._consume()
            node = _BinOp("^", node, self._power())
        return node

    def _unary(self) -> _ASTNode:
        if self._peek() == "-":
            self._consume()
            return _UnaryNeg(self._unary())
        return self._atom()

    def _atom(self) -> _ASTNode:
        token = self._peek()
        if token is None:
            raise SyntaxError("Unexpected end of expression in atom()")
        if token == "(":
            self._consume()
            node = self._expr()
            self._expect(")")
            return node
        if token.isdigit() or _is_rational_literal(token):
            self._consume()
            return _Literal(token)
        if token[0].isalpha() or token[0] == "_":
            self._consume()
            return _Var(token)
        raise SyntaxError(f"Unexpected token '{token}' at position {self.pos}")


_OP_MAP = {"+": ADD, "-": SUB, "*": MUL, "/": DIV, "^": POW}


def _ast_to_postfix(node: _ASTNode) -> List[str]:
    if isinstance(node, _Literal):
        return [node.value]
    if isinstance(node, _Var):
        return [node.name]
    if isinstance(node, _BinOp):
        return _ast_to_postfix(node.left) + _ast_to_postfix(node.right) + [_OP_MAP[node.op]]
    if isinstance(node, _UnaryNeg):
        return _ast_to_postfix(node.expr) + [NEG]
    raise TypeError(f"Unknown AST node: {type(node)}")


def infix_to_postfix(expr: str) -> List[str]:
    infix_tokens = _TOKEN_RE.findall(expr)
    if not infix_tokens:
        return []
    parser = _InfixParser(infix_tokens)
    return _ast_to_postfix(parser.parse())


def postfix_to_infix(tokens: Sequence[str]) -> str:
    op_sym = {ADD: "+", SUB: "-", MUL: "*", DIV: "/", POW: "^"}
    stack: list[str] = []
    for token in tokens:
        if token in BINARY_OPS:
            if len(stack) < 2:
                return "<INVALID: stack underflow>"
            rhs, lhs = stack.pop(), stack.pop()
            stack.append(f"({lhs} {op_sym[token]} {rhs})")
        elif token == NEG:
            if not stack:
                return "<INVALID: stack underflow>"
            stack.append(f"(-{stack.pop()})")
        else:
            stack.append(token)
    if len(stack) != 1:
        return f"<INVALID: stack has {len(stack)} elements>"
    return stack[0]


def token_category(token: str) -> str:
    if token in BINARY_OPS:
        return "binary_op"
    if token in UNARY_OPS:
        return "unary_op"
    if token in SPECIAL_TOKENS:
        return "special"
    if token == NUM_START:
        return "number_start"
    if token == NUM_END:
        return "number_end"
    if token in DIGIT_TOKENS:
        return "number_digit"
    return "operand"


def _encode_integer_literal(value: int) -> List[str]:
    digits = list(str(value))
    return [NUM_START] + [f"[DIGIT_{digit}]" for digit in digits] + [NUM_END]


def _collapse_numeric_tokens(tokens: Sequence[str]) -> List[str]:
    collapsed: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token != NUM_START:
            collapsed.append(token)
            index += 1
            continue
        index += 1
        digits: list[str] = []
        while index < len(tokens) and tokens[index] in DIGIT_TOKENS:
            digits.append(tokens[index][7:-1])
            index += 1
        if index >= len(tokens) or tokens[index] != NUM_END:
            collapsed.append("<INVALID_NUM>")
            break
        collapsed.append("".join(digits) if digits else "<INVALID_NUM>")
        index += 1
    return collapsed


class AmplitudeTokenizer:
    """Tokenizer for QED custom sequence targets."""

    def __init__(
        self,
        token2id: Optional[Dict[str, int]] = None,
        expression_mode: str = "postfix",
    ):
        if expression_mode not in {"postfix", "infix"}:
            raise ValueError("expression_mode must be either 'postfix' or 'infix'.")
        if token2id is not None:
            self.token2id = dict(token2id)
        else:
            self.token2id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.expression_mode = expression_mode
        self.min_postfix_tokens = 0
        self.max_postfix_tokens = 0
        self.avg_postfix_tokens = 0.0
        self._rebuild_reverse()

    def _rebuild_reverse(self) -> None:
        self.id2token = {value: key for key, value in self.token2id.items()}
        self.vocab_size = len(self.token2id)
        self.pad_id = self.token2id[PAD]
        self.sos_id = self.token2id[SOS]
        self.eos_id = self.token2id[EOS]
        self.unk_id = self.token2id[UNK]

    def build_vocab(self, expressions: Sequence[str]) -> "AmplitudeTokenizer":
        token_set: set[str] = set()
        if self.expression_mode == "postfix":
            token_set.update(BINARY_OPS)
            token_set.update(UNARY_OPS)
        else:
            token_set.update(INFIX_CONTROL_TOKENS)
        token_set.update(DEFAULT_SYMBOL_TOKENS)
        lengths: list[int] = []
        for expr in expressions:
            expr_tokens = self.tokenize_expr(expr)
            encoded_length = 0
            for token in expr_tokens:
                token_set.add(token)
                encoded_length += 1
            lengths.append(encoded_length)

        self.token2id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        for token in sorted(token_set):
            if token not in self.token2id:
                self.token2id[token] = len(self.token2id)

        if lengths:
            self.min_postfix_tokens = min(lengths)
            self.max_postfix_tokens = max(lengths)
            self.avg_postfix_tokens = sum(lengths) / len(lengths)
        else:
            self.min_postfix_tokens = 0
            self.max_postfix_tokens = 0
            self.avg_postfix_tokens = 0.0
        self._rebuild_reverse()
        return self

    @staticmethod
    def tokenize_infix_expr(expr: str) -> List[str]:
        return _INFIX_TOKEN_RE.findall(expr)

    def tokenize_expr(self, expr: str) -> List[str]:
        if self.expression_mode == "postfix":
            return infix_to_postfix(expr)
        return self.tokenize_infix_expr(expr)

    def encode_postfix_tokens(
        self,
        tokens: Sequence[str],
        add_sos: bool = True,
        add_eos: bool = True,
        max_len: Optional[int] = None,
    ) -> List[int]:
        ids: list[int] = []
        if add_sos:
            ids.append(self.sos_id)
        for token in tokens:
            ids.append(self.token2id.get(token, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        if max_len is not None and len(ids) > max_len:
            ids = ids[: max_len - 1] + [self.eos_id]
        return ids

    def encode(
        self,
        expr: str,
        add_sos: bool = True,
        add_eos: bool = True,
        max_len: Optional[int] = None,
    ) -> List[int]:
        return self.encode_postfix_tokens(
            self.tokenize_expr(expr),
            add_sos=add_sos,
            add_eos=add_eos,
            max_len=max_len,
        )

    def encode_tensor(
        self,
        expr: str,
        max_len: int = 1700,
        add_sos: bool = True,
        add_eos: bool = True,
    ) -> Tensor:
        ids = self.encode(expr, add_sos=add_sos, add_eos=add_eos, max_len=max_len)
        out = torch.full((max_len,), self.pad_id, dtype=torch.long)
        out[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out

    def decode_tokens(self, ids: Sequence[int], strip_special: bool = True) -> List[str]:
        tokens: list[str] = []
        for index in ids:
            token = self.id2token.get(int(index), UNK)
            if strip_special and token in (PAD, SOS, EOS):
                continue
            tokens.append(token)
        return tokens

    def decode(self, ids: Sequence[int], strip_special: bool = True) -> str:
        return " ".join(self.decode_tokens(ids, strip_special=strip_special))

    def decode_to_infix(self, ids: Sequence[int]) -> str:
        decoded_tokens = self.decode_tokens(ids)
        if self.expression_mode == "infix":
            return " ".join(decoded_tokens)
        return postfix_to_infix(decoded_tokens)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(
                {
                    "token2id": self.token2id,
                    "expression_mode": self.expression_mode,
                    "min_postfix_tokens": self.min_postfix_tokens,
                    "max_postfix_tokens": self.max_postfix_tokens,
                    "avg_postfix_tokens": self.avg_postfix_tokens,
                },
                handle,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "AmplitudeTokenizer":
        with open(path) as handle:
            payload = json.load(handle)
        tokenizer = cls(
            token2id=payload["token2id"],
            expression_mode=payload.get("expression_mode", "postfix"),
        )
        tokenizer.min_postfix_tokens = int(payload.get("min_postfix_tokens", 0))
        tokenizer.max_postfix_tokens = int(payload.get("max_postfix_tokens", 0))
        tokenizer.avg_postfix_tokens = float(payload.get("avg_postfix_tokens", 0.0))
        tokenizer._rebuild_reverse()
        return tokenizer


FactorizedTokenizer = AmplitudeTokenizer
