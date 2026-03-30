"""Training utilities for the QED amplitude -> squared-amplitude package."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from .config import QEDAmp2SqConfig
from .dataset import (
    QEDAmp2SqBatch,
    QEDAmp2SqDataset,
    build_source_tokenizer,
    collate_examples,
    load_qed_amp2sq_corpus,
    resolve_splits,
)
from .factorization import (
    TARGET_VARIANT_RAW_STRING,
    default_decoder_representation,
    default_use_grammar,
    normalize_target_variant,
)
from .grammar import RPNGrammar
from .model import QEDAmp2SqModel
from .tokenizer import AmplitudeTokenizer


@torch.no_grad()
def token_accuracy(logits: Tensor, targets: Tensor, pad_id: int) -> float:
    preds = logits.argmax(dim=-1)
    mask = targets != pad_id
    correct = (preds == targets) & mask
    return correct.sum().item() / max(mask.sum().item(), 1)


@torch.no_grad()
def sequence_match(
    pred_ids: Tensor,
    target_ids: Tensor,
    pad_id: int,
    eos_id: int,
) -> Tensor:
    matches = []
    for idx in range(target_ids.size(0)):
        target = []
        for tok in target_ids[idx].tolist()[1:]:
            if tok in (pad_id, eos_id):
                break
            target.append(tok)

        pred = []
        for tok in pred_ids[idx].tolist()[1:]:
            if tok in (pad_id, eos_id):
                break
            pred.append(tok)
        matches.append(pred == target)
    return torch.tensor(matches, dtype=torch.bool, device=target_ids.device)


class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs: int, base_scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.current_epoch = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, metrics: float | None = None):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            scale = self.current_epoch / max(self.warmup_epochs, 1)
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr * scale
            return
        if isinstance(self.base_scheduler, ReduceLROnPlateau):
            self.base_scheduler.step(metrics)
        else:
            self.base_scheduler.step()


class QEDAmp2SqTrainer:
    def __init__(self, config: QEDAmp2SqConfig):
        self.config = config
        self.target_variant = normalize_target_variant(config.data.target_variant)
        self.config.data.target_variant = self.target_variant
        self.encoder_variant = config.model.encoder_variant
        if config.model.decoder_representation == "auto":
            config.model.decoder_representation = default_decoder_representation(
                self.target_variant
            )
        if config.model.use_grammar is None:
            config.model.use_grammar = default_use_grammar(self.target_variant)
        if (
            config.model.decoder_representation != "postfix"
            and bool(config.model.use_grammar)
        ):
            raise ValueError(
                "Grammar-constrained decoding is only supported for postfix targets."
            )
        self.device = config.resolve_device()
        print(f"Using device: {self.device}")

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.output_dir / "cache" / "qed_amp2sq_compiled_v1"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.corpus = load_qed_amp2sq_corpus(
            data_dir=config.data.data_dir,
            cache_dir=self.cache_dir,
            target_variant=config.data.target_variant,
        )
        (
            self.train_indices,
            self.val_indices,
            self.test_indices,
        ) = resolve_splits(
            self.corpus,
            seed=config.training.seed,
        )

        self.source_tokenizer = build_source_tokenizer(
            self.corpus,
            self.train_indices,
            self.encoder_variant,
        )
        train_target_sequences = [
            self.corpus.targets[index].sequence_target_text()
            for index in self.train_indices
        ]
        self.tokenizer = AmplitudeTokenizer(
            expression_mode=config.model.decoder_representation
        ).build_vocab(train_target_sequences)
        self.source_tokenizer.save(self.output_dir / "source_tokenizer.json")
        self.tokenizer.save(self.output_dir / "tokenizer.json")

        self.train_loader = self._build_loader(self.train_indices, shuffle=True)
        self.val_loader = self._build_loader(self.val_indices, shuffle=False)
        self.test_loader = self._build_loader(self.test_indices, shuffle=False)

        self.grammar = None
        if bool(config.model.use_grammar):
            self.grammar = RPNGrammar(self.tokenizer).to(self.device)

        self.model = QEDAmp2SqModel(
            vocab_size=self.tokenizer.vocab_size,
            node_in_dim=0,
            src_vocab_size=self.source_tokenizer.vocab_size,
            src_pad_id=self.source_tokenizer.pad_id,
            hidden_dim=config.model.hidden_dim,
            stream_dim=config.model.stream_dim,
            graph_dim=config.model.graph_dim,
            num_mp_layers=config.model.num_mp_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            dec_d_model=config.model.dec_d_model,
            dec_nhead=config.model.dec_nhead,
            dec_layers=config.model.dec_layers,
            dec_dim_ff=config.model.dec_dim_ff,
            max_tgt_seq_len=config.model.max_tgt_seq_len,
            max_src_term_len=config.model.max_src_term_len,
            max_flat_src_len=config.model.max_flat_src_len,
            pad_id=self.tokenizer.pad_id,
            encoder_variant=config.model.encoder_variant,
            target_variant=config.data.target_variant,
            use_grammar=config.model.use_grammar,
        ).to(self.device)

        self.sequence_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        if config.training.scheduler == "cosine":
            base_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(config.training.max_epochs - config.training.warmup_epochs, 1),
                eta_min=config.training.min_lr,
            )
        elif config.training.scheduler == "plateau":
            base_scheduler = ReduceLROnPlateau(
                self.optimizer,
                patience=10,
                factor=0.5,
                min_lr=config.training.min_lr,
            )
        else:
            raise ValueError(f"Unknown scheduler: {config.training.scheduler}")

        self.scheduler = WarmupScheduler(
            self.optimizer,
            config.training.warmup_epochs,
            base_scheduler,
        )

        self.best_val_exact = 0.0
        self.best_val_loss = float("inf")
        self.select_best_by_loss = False
        self.decode_during_training = True
        self.eval_checkpoint_name = "best.pt"
        self.stop_reason: str | None = None
        self.best_val_epoch: int | None = None
        self.first_val_exact_epoch: int | None = None
        self.history: Dict[str, list[float | None]] = {
            "train_loss": [],
            "train_seq_accuracy": [],
            "val_loss": [],
            "val_seq_accuracy": [],
            "lr": [],
            "time": [],
        }

    def _build_loader(self, indices: list[int], shuffle: bool) -> DataLoader:
        dataset = QEDAmp2SqDataset(
            corpus=self.corpus,
            indices=indices,
            source_tokenizer=self.source_tokenizer,
            target_tokenizer=self.tokenizer,
            max_src_term_len=self.config.model.max_src_term_len,
            max_src_terms=self.config.model.max_src_terms,
            max_flat_src_len=self.config.model.max_flat_src_len,
            max_tgt_seq_len=self.config.model.max_tgt_seq_len,
            encoder_variant=self.encoder_variant,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=self.config.training.num_workers,
            collate_fn=collate_examples,
        )

    def _loss_and_metrics(
        self,
        batch: QEDAmp2SqBatch,
        train_mode: bool,
        decode_sequences: bool | None = None,
    ) -> tuple[Tensor, dict[str, float | None]]:
        if decode_sequences is None:
            decode_sequences = not train_mode
        tokens = batch.tgt_tokens
        tokens_in = tokens[:, :-1]
        tokens_out = tokens[:, 1:]
        outputs = self.model(batch, tgt_tokens=tokens_in)
        sample_ids = batch.sample_idx.detach().cpu().tolist()
        if not torch.isfinite(outputs.sequence_logits).all():
            raise RuntimeError(
                f"Non-finite decoder logits encountered for sample_idx={sample_ids}."
            )
        loss = self.sequence_criterion(
            outputs.sequence_logits.reshape(-1, outputs.sequence_logits.size(-1)),
            tokens_out.reshape(-1),
        )
        if not torch.isfinite(loss):
            mode = "train" if train_mode else "eval"
            raise RuntimeError(
                f"Non-finite {mode} loss encountered for sample_idx={sample_ids}."
            )

        metrics = {
            "token_acc": token_accuracy(
                outputs.sequence_logits,
                tokens_out,
                self.tokenizer.pad_id,
            ),
        }
        if decode_sequences:
            generated = self.model.generate(
                batch,
                max_len=self.config.model.max_gen_len,
                sos_id=self.tokenizer.sos_id,
                eos_id=self.tokenizer.eos_id,
                grammar=self.grammar,
            )
            seq_match = sequence_match(
                generated.sequence_ids,
                tokens,
                self.tokenizer.pad_id,
                self.tokenizer.eos_id,
            )
            generated_lengths = []
            for row in generated.sequence_ids.tolist():
                length = 0
                for tok in row[1:]:
                    if tok in (self.tokenizer.pad_id, self.tokenizer.eos_id):
                        break
                    length += 1
                generated_lengths.append(length)
            metrics["seq_exact"] = seq_match.float().mean().item()
            metrics["exact"] = metrics["seq_exact"]
            metrics["generated_len"] = (
                sum(generated_lengths) / max(len(generated_lengths), 1)
            )
        else:
            metrics["seq_exact"] = None
            metrics["exact"] = None
            metrics["generated_len"] = None
        return loss, metrics

    @staticmethod
    def _assert_finite_metric_dict(
        metrics: dict[str, float | None],
        *,
        split_name: str,
    ) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Non-finite metric detected for split='{split_name}' "
                    f"key='{key}' value={value!r}."
                )

    @staticmethod
    def _should_log_epoch(epoch: int, improved: bool, log_interval: int) -> bool:
        del improved
        return epoch <= 5 or epoch % log_interval == 0

    def _should_decode_epoch(self, epoch: int) -> bool:
        if self.target_variant != TARGET_VARIANT_RAW_STRING:
            return True
        interval = max(int(self.config.training.val_every_n_epochs), 1)
        return epoch % interval == 0 or epoch == self.config.training.max_epochs

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_token_acc = 0.0
        total_examples = 0

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            try:
                loss, metrics = self._loss_and_metrics(batch, train_mode=True)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"[train] epoch={epoch} batch={batch_idx}: {exc}"
                ) from exc
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss encountered.")
            loss.backward()
            if self.config.training.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )
            self.optimizer.step()

            batch_size = int(batch.tgt_tokens.size(0))
            total_loss += loss.item() * batch_size
            total_token_acc += metrics["token_acc"] * batch_size
            total_examples += batch_size

        denom = max(total_examples, 1)
        return total_loss / denom, total_token_acc / denom

    @torch.no_grad()
    def evaluate_loader(
        self,
        loader: DataLoader,
        *,
        split_name: str,
        epoch: int,
        decode_sequences: bool = True,
    ) -> dict[str, float | None]:
        self.model.eval()
        totals = {
            "loss": 0.0,
            "token_acc": 0.0,
            "seq_exact": 0.0,
            "exact": 0.0,
            "generated_len": 0.0,
        }
        total_examples = 0
        total_generated_examples = 0

        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(self.device)
            try:
                loss, metrics = self._loss_and_metrics(
                    batch,
                    train_mode=False,
                    decode_sequences=decode_sequences,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"[{split_name}] epoch={epoch} batch={batch_idx}: {exc}"
                ) from exc
            batch_size = int(batch.tgt_tokens.size(0))
            totals["loss"] += loss.item() * batch_size
            totals["token_acc"] += metrics["token_acc"] * batch_size
            if metrics["seq_exact"] is not None:
                totals["seq_exact"] += metrics["seq_exact"] * batch_size
                totals["exact"] += metrics["exact"] * batch_size
                totals["generated_len"] += metrics["generated_len"] * batch_size
                total_generated_examples += batch_size
            total_examples += batch_size

        denom = max(total_examples, 1)
        averaged: dict[str, float | None] = {
            "loss": totals["loss"] / denom,
            "token_acc": totals["token_acc"] / denom,
            "seq_exact": None,
            "exact": None,
            "generated_len": None,
        }
        if total_generated_examples > 0:
            averaged["seq_exact"] = totals["seq_exact"] / total_generated_examples
            averaged["exact"] = totals["exact"] / total_generated_examples
            averaged["generated_len"] = (
                totals["generated_len"] / total_generated_examples
            )
        self._assert_finite_metric_dict(averaged, split_name=split_name)
        return averaged

    def train(self) -> Dict[str, object]:
        cfg = self.config
        print(f"\n{'=' * 60}")
        print(f"  QED Amp2Sq Training: {cfg.experiment_name}")
        print(f"{'=' * 60}")
        print(f"  Encoder: {cfg.model.encoder_variant}")
        print(
            "  Source view: "
            + (
                "physics-tagged QED amplitude sequence"
                if self.encoder_variant == "custom"
                else "raw amplitude token baseline"
            )
        )
        print(f"  Target variant: {self.target_variant}")
        print(f"  Decoder representation: {cfg.model.decoder_representation}")
        print(f"  Grammar constrained: {cfg.model.use_grammar}")
        print(
            f"  Train/val/test: {len(self.train_indices)}/{len(self.val_indices)}/{len(self.test_indices)}"
        )
        print(f"  Epochs: {cfg.training.max_epochs}")
        print(f"  Batch size: {cfg.training.batch_size}")
        print(f"  Learning rate: {cfg.training.learning_rate}")
        print(f"  Device: {self.device}")
        print(f"{'=' * 60}\n")

        start_time = time.time()
        epochs_completed = 0
        log_interval = max(int(cfg.training.log_every_n_steps), 1)
        for epoch in range(1, cfg.training.max_epochs + 1):
            train_loss, _ = self.train_epoch(epoch)
            decode_this_epoch = self._should_decode_epoch(epoch)

            val_metrics = self.evaluate_loader(
                self.val_loader,
                split_name="val",
                epoch=epoch,
                decode_sequences=decode_this_epoch,
            )
            if val_metrics["seq_exact"] is None:
                improved = False
            else:
                improved = val_metrics["seq_exact"] > self.best_val_exact or (
                    val_metrics["seq_exact"] == self.best_val_exact
                    and val_metrics["loss"] < self.best_val_loss
                )
            if improved:
                if val_metrics["seq_exact"] is not None:
                    self.best_val_exact = val_metrics["seq_exact"]
                self.best_val_loss = val_metrics["loss"]
                self.best_val_epoch = epoch
                self.eval_checkpoint_name = "best.pt"
                self._save_checkpoint("best.pt", epoch)

            if (
                decode_this_epoch
                and self.first_val_exact_epoch is None
                and val_metrics["seq_exact"] is not None
                and val_metrics["seq_exact"] >= 1.0
            ):
                self.first_val_exact_epoch = epoch

            self.scheduler.step(val_metrics["loss"])
            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            epochs_completed = epoch

            self.history["train_loss"].append(train_loss)
            self.history["train_seq_accuracy"].append(None)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_seq_accuracy"].append(val_metrics["seq_exact"])
            self.history["lr"].append(current_lr)
            self.history["time"].append(elapsed)

            if self._should_log_epoch(epoch, improved, log_interval):
                if decode_this_epoch and val_metrics["seq_exact"] is not None:
                    log = (
                        f"Epoch {epoch:04d} | "
                        f"train {train_loss:.4f} | "
                        f"val {val_metrics['loss']:.4f} / seq {val_metrics['seq_exact']:.3f}"
                    )
                else:
                    log = (
                        f"Epoch {epoch:04d} | train {train_loss:.4f} | "
                        f"val {val_metrics['loss']:.4f}"
                    )
                if improved:
                    log += " | improved"
                log += f" | {elapsed:.1f}s"
                print(log)

            if (
                decode_this_epoch
                and cfg.training.stop_on_val_exact is not None
                and val_metrics["seq_exact"] is not None
                and val_metrics["seq_exact"] >= cfg.training.stop_on_val_exact
            ):
                self.stop_reason = (
                    f"Stopped at epoch {epoch} because val_seq_accuracy "
                    f"reached {val_metrics['seq_exact']:.3f}."
                )
                print(self.stop_reason)
                break

        self._save_checkpoint("final.pt", epochs_completed)
        history_payload = dict(self.history)
        history_payload["epochs_ran"] = epochs_completed
        history_payload["stopped_early"] = self.stop_reason is not None
        history_payload["stop_reason"] = self.stop_reason
        history_payload["best_val_epoch"] = self.best_val_epoch
        history_payload["first_val_exact_epoch"] = self.first_val_exact_epoch
        history_payload["selection_metric"] = (
            "val_seq_accuracy_every_n_epochs_then_val_loss"
            if self.target_variant == TARGET_VARIANT_RAW_STRING
            else "val_seq_accuracy_then_val_loss"
        )
        with open(self.output_dir / "training_history.json", "w") as handle:
            json.dump(history_payload, handle, indent=2)

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Best val seq accuracy: {self.best_val_exact:.4f}")
        print(f"Best val loss:  {self.best_val_loss:.6f}")
        if self.best_val_epoch is not None:
            print(f"Best val epoch: {self.best_val_epoch}")
        if self.first_val_exact_epoch is not None:
            print(
                f"First val seq accuracy=1.0 epoch: {self.first_val_exact_epoch}"
            )
        return history_payload

    def _save_checkpoint(self, filename: str, epoch: int) -> None:
        path = self.output_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_exact": self.best_val_exact,
                "best_val_loss": self.best_val_loss,
                "config": asdict(self.config),
                "tokenizer": {
                    "token2id": self.tokenizer.token2id,
                    "expression_mode": self.tokenizer.expression_mode,
                },
                "source_tokenizer": {"token2id": self.source_tokenizer.token2id},
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_exact = float(checkpoint.get("best_val_exact", 0.0))
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))

    def evaluate(self) -> Dict[str, float]:
        checkpoint_path = self.output_dir / self.eval_checkpoint_name
        if not checkpoint_path.exists():
            checkpoint_path = self.output_dir / "best.pt"
        if checkpoint_path.exists():
            self.load_checkpoint(checkpoint_path)
        metrics = self.evaluate_loader(
            self.test_loader,
            split_name="test_final",
            epoch=0,
            decode_sequences=True,
        )
        metrics = {
            "test_loss": metrics["loss"],
            "test_token_accuracy": metrics["token_acc"],
            "test_seq_accuracy": metrics["seq_exact"],
            "test_exact": metrics["exact"],
            "test_generated_len": metrics["generated_len"],
        }
        print("\nTest Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
        return metrics
