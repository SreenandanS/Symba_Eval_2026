"""CLI runner for the QCD custom fd-to-squared-amplitude package."""

from __future__ import annotations

import argparse

import torch

from .factorization import (
    TARGET_VARIANT_RAW_STRING,
    default_decoder_representation,
    default_use_grammar,
    normalize_target_variant,
)


def smoke_test(
    data_dir: str = ".",
    *,
    encoder_variant: str = "custom",
    target_variant: str = "factorized",
):
    print("=" * 60)
    print("  Custom Encoder Smoke Test")
    print("=" * 60)

    from .runtime import ensure_torch_geometric

    ensure_torch_geometric(auto_install=False)
    from .factorization import factorize_squared_amplitude
    from .grammar import RPNGrammar
    from .model import CustomQCDFd2SqModel
    from .parser import parse_all_qcd
    from .tokenizer import AmplitudeTokenizer

    target_variant = normalize_target_variant(target_variant)
    decoder_representation = default_decoder_representation(target_variant)
    use_grammar = default_use_grammar(target_variant)

    print("\n[1/6] Parsing QCD data...")
    diagrams = parse_all_qcd(data_dir)
    print(f"  Parsed {len(diagrams)} diagrams")

    print("\n[2/6] Building targets...")
    targets = [
        factorize_squared_amplitude(
            diagram.raw_squared,
            target_variant=target_variant,
        )
        for diagram in diagrams[:5]
    ]
    print(f"  Sample target: {targets[0].sequence_target_text()[:120]}")

    print("\n[3/6] Building tokenizer...")
    tokenizer = AmplitudeTokenizer(expression_mode=decoder_representation).build_vocab(
        [target.sequence_target_text() for target in targets]
    )
    grammar = RPNGrammar(tokenizer) if use_grammar else None
    print(f"  Target vocab: {tokenizer.vocab_size}")

    print("\n[4/6] Building graphs...")
    from .feynman_graph import diagram_to_homogeneous_graph

    max_seq_len = 256
    graphs = [diagram_to_homogeneous_graph(diagram) for diagram in diagrams[:5]]
    for graph, target in zip(graphs, targets):
        graph.tgt_tokens = tokenizer.encode_tensor(
            target.sequence_target_text(),
            max_len=max_seq_len,
        ).unsqueeze(0)

    print("\n[5/6] Building model...")
    from torch_geometric.data import Batch

    batch = Batch.from_data_list(graphs)
    node_dim = graphs[0].x.size(1)
    edge_dim = graphs[0].edge_attr.size(1) if graphs[0].edge_attr.size(0) > 0 else 15
    model = CustomQCDFd2SqModel(
        vocab_size=tokenizer.vocab_size,
        node_in_dim=node_dim,
        edge_in_dim=edge_dim,
        hidden_dim=64,
        stream_dim=32,
        graph_dim=64,
        num_mp_layers=2,
        num_heads=2,
        dec_d_model=64,
        dec_nhead=2,
        dec_layers=2,
        dec_dim_ff=128,
        max_seq_len=max_seq_len,
        pad_id=tokenizer.pad_id,
        encoder_variant=encoder_variant,
        target_variant=target_variant,
        use_grammar=use_grammar,
    )
    print(f"  Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("\n[6/6] Running forward + generation...")
    tokens = batch.tgt_tokens
    tokens_in = tokens[:, :-1]
    with torch.no_grad():
        outputs = model(batch, tgt_sequence=tokens_in)
        generated = model.generate(
            batch,
            max_len=64,
            sos_id=tokenizer.sos_id,
            eos_id=tokenizer.eos_id,
            grammar=grammar,
        )
    print(f"  Decoder logits: {tuple(outputs.sequence_logits.shape)}")
    print(f"  Pred sequence shape: {tuple(generated.sequence_ids.shape)}")
    print("=" * 60)
    print("  Smoke test PASSED")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Custom encoder for QCD Feynman diagrams",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs/custom_qcd_fd2sq")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--target-variant",
        type=str,
        choices=("factorized", "raw_string"),
        default="factorized",
    )
    parser.add_argument(
        "--encoder-variant",
        type=str,
        choices=("custom", "seq2seq"),
        default=None,
    )
    parser.add_argument("--no-grammar", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(
            data_dir=args.data_dir,
            encoder_variant=args.encoder_variant or "custom",
            target_variant=args.target_variant,
        )
        return

    from .config import CustomQCDFd2SqConfig
    from .runtime import ensure_torch_geometric
    from .train import CustomQCDFd2SqTrainer

    ensure_torch_geometric(auto_install=False)

    config = CustomQCDFd2SqConfig()
    if args.config:
        import yaml

        with open(args.config) as handle:
            cfg_dict = yaml.safe_load(handle)
        if "model" in cfg_dict:
            for key, value in cfg_dict["model"].items():
                setattr(config.model, key, value)
        if "training" in cfg_dict:
            for key, value in cfg_dict["training"].items():
                setattr(config.training, key, value)
        if "data" in cfg_dict:
            for key, value in cfg_dict["data"].items():
                setattr(config.data, key, value)

    target_variant = normalize_target_variant(args.target_variant)
    config.data.data_dir = args.data_dir
    config.data.target_variant = target_variant
    config.output_dir = args.output_dir
    config.device = args.device

    if args.epochs:
        config.training.max_epochs = args.epochs
    else:
        config.training.max_epochs = (
            500 if target_variant == TARGET_VARIANT_RAW_STRING else 250
        )
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.encoder_variant:
        config.model.encoder_variant = args.encoder_variant
    if args.no_grammar:
        config.model.use_grammar = False

    trainer = CustomQCDFd2SqTrainer(config)
    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()
