"""Dataclass configuration for canonical amplitude -> squared-amplitude experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    node_in_dim: int = 48
    hidden_dim: int = 192
    stream_dim: int = 128
    graph_dim: int = 192
    num_mp_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    dec_d_model: int = 384
    dec_nhead: int = 8
    dec_layers: int = 6
    dec_dim_ff: int = 1024

    max_src_term_len: int = 192
    max_src_terms: int = 32
    max_flat_src_len: int = 3072
    max_tgt_seq_len: int = 1400
    max_gen_len: int = 2048

    encoder_variant: str = "custom"
    decoder_representation: str = "auto"
    use_grammar: bool | None = None


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_epochs: int = 200
    warmup_epochs: int = 10
    min_lr: float = 1e-6

    scheduler: str = "cosine"

    batch_size: int = 16
    gradient_clip: float = 1.0

    seed: int = 42
    num_workers: int = 0
    log_every_n_steps: int = 10
    val_every_n_epochs: int = 20
    stop_on_val_exact: float | None = None
    stop_on_test_exact: float | None = None


@dataclass
class DataConfig:
    data_dir: str = "."
    target_variant: str = "factorized"


@dataclass
class Amp2SqConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "outputs/custom_qcd_amp2sq"
    experiment_name: str = "qcd_amp_to_sq"
    device: str = "auto"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
