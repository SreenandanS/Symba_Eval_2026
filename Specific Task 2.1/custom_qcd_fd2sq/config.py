"""Dataclass configuration for the QCD custom fd-to-squared-amplitude experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    node_in_dim: int = 19
    edge_in_dim: int = 15
    hidden_dim: int = 128
    stream_dim: int = 64
    graph_dim: int = 128
    num_mp_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1

    dec_d_model: int = 256
    dec_nhead: int = 4
    dec_layers: int = 4
    dec_dim_ff: int = 512
    max_seq_len: int = 1400
    max_src_seq_len: int = 256
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

    batch_size: int = 32
    gradient_clip: float = 1.0

    seed: int = 42
    num_workers: int = 0
    log_every_n_steps: int = 10
    val_every_n_epochs: int = 1
    stop_on_val_exact: float | None = None
    stop_on_test_exact: float | None = None


@dataclass
class DataConfig:
    data_dir: str = "."
    target_variant: str = "factorized"
    augment_kinematics: bool = False
    num_kinematic_samples: int = 1


@dataclass
class CustomQCDFd2SqConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "outputs/custom_qcd_fd2sq"
    experiment_name: str = "qcd_tree_2to2"
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
