"""CLI runner for the fixed-slot QED custom fd-to-squared-amplitude package."""

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
    print("  QED Custom Encoder Smoke Test")
    print("=" * 60)

    from .config import CustomQEDFd2SqConfig
    from .dataset import create_dataloaders
    from .grammar import RPNGrammar
    from .model import CustomQEDFd2SqModel

    target_variant = normalize_target_variant(target_variant)
    decoder_representation = default_decoder_representation(target_variant)
    use_grammar = default_use_grammar(target_variant)

    cfg = CustomQEDFd2SqConfig()

    print("\n[1/4] Building dataloaders...")
    train_loader, _val_loader, _test_loader, tokenizer = create_dataloaders(
        data_dir=data_dir,
        batch_size=4,
        max_seq_len=256,
        seed=cfg.training.seed,
        expression_mode=decoder_representation,
        target_variant=target_variant,
    )
    batch = next(iter(train_loader))
    grammar = RPNGrammar(tokenizer) if use_grammar else None
    print(f"  Batch slot tensor: {tuple(batch.x.shape)}")
    print(f"  Target vocab: {tokenizer.vocab_size}")

    print("\n[2/4] Building model...")
    model = CustomQEDFd2SqModel(
        vocab_size=tokenizer.vocab_size,
        node_in_dim=batch.x.size(-1),
        hidden_dim=64,
        stream_dim=32,
        slot_dim=32,
        num_mp_layers=2,
        num_heads=2,
        dec_d_model=64,
        dec_nhead=2,
        dec_layers=2,
        dec_dim_ff=128,
        max_seq_len=256,
        pad_id=tokenizer.pad_id,
        encoder_variant=encoder_variant,
        target_variant=target_variant,
        use_grammar=use_grammar,
    )
    print(f"  Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("\n[3/4] Running forward...")
    tokens = batch.tgt_tokens
    tokens_in = tokens[:, :-1]
    with torch.no_grad():
        outputs = model(batch, tgt_sequence=tokens_in)
    print(f"  Decoder logits: {tuple(outputs.sequence_logits.shape)}")

    print("\n[4/4] Running generation...")
    with torch.no_grad():
        generated = model.generate(
            batch,
            max_len=64,
            sos_id=tokenizer.sos_id,
            eos_id=tokenizer.eos_id,
            grammar=grammar,
        )
    print(f"  Pred sequence shape: {tuple(generated.sequence_ids.shape)}")
    print("=" * 60)
    print("  Smoke test PASSED")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Custom encoder for tree-level 2->2 QED diagrams",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs/custom_qed_fd2sq")
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

    from .config import CustomQEDFd2SqConfig
    from .train import CustomQEDFd2SqTrainer

    config = CustomQEDFd2SqConfig()
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
            500 if target_variant == TARGET_VARIANT_RAW_STRING else 300
        )
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.encoder_variant:
        config.model.encoder_variant = args.encoder_variant
    if args.no_grammar:
        config.model.use_grammar = False

    trainer = CustomQEDFd2SqTrainer(config)
    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()
