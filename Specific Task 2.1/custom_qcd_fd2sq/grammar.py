"""Local postfix grammar for squared-amplitude decoder targets."""

from __future__ import annotations

import torch
from torch import Tensor

from .tokenizer import (
    AmplitudeTokenizer,
    DIGIT_TOKENS,
    NUM_END,
    NUM_START,
    token_category,
)


class RPNGrammar:
    def __init__(self, tokenizer: AmplitudeTokenizer):
        self.tokenizer = tokenizer
        vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.pad_id
        self.eos_id = tokenizer.eos_id
        self.num_start_id = tokenizer.token2id.get(NUM_START)
        self.num_end_id = tokenizer.token2id.get(NUM_END)
        digit_ids = [
            tokenizer.token2id[token]
            for token in DIGIT_TOKENS
            if token in tokenizer.token2id
        ]
        self.digit_ids = torch.tensor(digit_ids, dtype=torch.long)
        self.min_content_tokens_before_eos = max(
            int(getattr(tokenizer, "min_postfix_tokens", 0)),
            0,
        )

        self._delta_outside_number = torch.zeros(vocab_size, dtype=torch.long)
        self._is_content = torch.zeros(vocab_size, dtype=torch.bool)
        self._normal_masks = torch.zeros(3, vocab_size, dtype=torch.bool)
        self._number_masks_no_digit = torch.zeros(vocab_size, dtype=torch.bool)
        self._number_masks_with_digit = torch.zeros(vocab_size, dtype=torch.bool)

        for token, token_id in tokenizer.token2id.items():
            category = token_category(token)
            if category != "special":
                self._is_content[token_id] = True

            if category == "operand":
                self._delta_outside_number[token_id] = 1
                self._normal_masks[0, token_id] = True
                self._normal_masks[1, token_id] = True
                self._normal_masks[2, token_id] = True
            elif category == "binary_op":
                self._delta_outside_number[token_id] = -1
                self._normal_masks[2, token_id] = True
            elif category == "unary_op":
                self._normal_masks[1, token_id] = True
                self._normal_masks[2, token_id] = True
            elif category == "number_start":
                self._normal_masks[0, token_id] = True
                self._normal_masks[1, token_id] = True
                self._normal_masks[2, token_id] = True
            elif category == "number_digit":
                self._number_masks_no_digit[token_id] = True
                self._number_masks_with_digit[token_id] = True
            elif category == "number_end":
                self._number_masks_with_digit[token_id] = True

        self._normal_masks[1, tokenizer.eos_id] = True

    def get_valid_mask(
        self,
        stack_depths: Tensor,
        generated_content_lengths: Tensor | None = None,
        in_number: Tensor | None = None,
        number_has_digit: Tensor | None = None,
    ) -> Tensor:
        if in_number is None:
            in_number = torch.zeros_like(stack_depths, dtype=torch.bool)
        if number_has_digit is None:
            number_has_digit = torch.zeros_like(stack_depths, dtype=torch.bool)

        buckets = torch.clamp(stack_depths, max=2)
        valid = self._normal_masks[buckets].clone()

        if in_number.any():
            no_digit_rows = in_number & (~number_has_digit)
            with_digit_rows = in_number & number_has_digit
            if no_digit_rows.any():
                valid[no_digit_rows] = self._number_masks_no_digit
            if with_digit_rows.any():
                valid[with_digit_rows] = self._number_masks_with_digit

        if (
            generated_content_lengths is not None
            and self.min_content_tokens_before_eos > 0
        ):
            too_short = generated_content_lengths < self.min_content_tokens_before_eos
            if too_short.any():
                valid[too_short, self.eos_id] = False
        return valid

    def batch_transition(
        self,
        token_ids: Tensor,
        in_number: Tensor,
        number_has_digit: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        next_in_number = in_number.clone()
        next_number_has_digit = number_has_digit.clone()
        delta = torch.zeros_like(token_ids, dtype=torch.long)

        outside = ~in_number
        if self.num_start_id is None:
            start_number = torch.zeros_like(outside)
        else:
            start_number = outside & (token_ids == self.num_start_id)
        next_in_number[start_number] = True
        next_number_has_digit[start_number] = False

        normal_tokens = outside & (~start_number)
        delta[normal_tokens] = self._delta_outside_number[token_ids[normal_tokens]]

        inside = in_number
        digit_mask = inside & self._is_digit_token(token_ids)
        next_number_has_digit[digit_mask] = True

        if self.num_end_id is None:
            end_number = torch.zeros_like(inside)
        else:
            end_number = inside & (token_ids == self.num_end_id)
        delta[end_number] = 1
        next_in_number[end_number] = False
        next_number_has_digit[end_number] = False
        return delta, next_in_number, next_number_has_digit

    def batch_is_content(self, token_ids: Tensor) -> Tensor:
        return self._is_content[token_ids]

    def _is_digit_token(self, token_ids: Tensor) -> Tensor:
        if self.digit_ids.numel() == 0:
            return torch.zeros_like(token_ids, dtype=torch.bool)
        return (token_ids.unsqueeze(-1) == self.digit_ids.to(token_ids.device)).any(dim=-1)

    def to(self, device: torch.device | str) -> "RPNGrammar":
        self.digit_ids = self.digit_ids.to(device)
        self._delta_outside_number = self._delta_outside_number.to(device)
        self._is_content = self._is_content.to(device)
        self._normal_masks = self._normal_masks.to(device)
        self._number_masks_no_digit = self._number_masks_no_digit.to(device)
        self._number_masks_with_digit = self._number_masks_with_digit.to(device)
        return self
