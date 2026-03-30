"""CLI runner for the QED amplitude -> squared-amplitude package."""

from __future__ import annotations

import argparse
import json
import torch

from .factorization import (
    TARGET_VARIANT_RAW_STRING,
    default_decoder_representation,
    default_use_grammar,
    normalize_target_variant,
)


def smoke_test(
    data_dir: str = ".",
    encoder_variant: str = "custom",
    batch_size: int = 2,
    target_variant: str = "factorized",
):
    print("=" * 60)
    print("  QED Amp2Sq Smoke Test")
    print("=" * 60)

    from .dataset import (
        QEDAmp2SqDataset,
        build_source_tokenizer,
        collate_examples,
        load_qed_amp2sq_corpus,
        resolve_splits,
    )
    from .grammar import RPNGrammar
    from .model import QEDAmp2SqModel
    from .tokenizer import AmplitudeTokenizer

    target_variant = normalize_target_variant(target_variant)
    decoder_representation = default_decoder_representation(target_variant)
    use_grammar = default_use_grammar(target_variant)

    corpus = load_qed_amp2sq_corpus(
        data_dir,
        cache_dir="outputs/custom_qed_amp2sq/smoke_cache",
        target_variant=target_variant,
    )
    print(f"  Parsed {len(corpus.targets)} amplitude/squared-amplitude pairs")

    train_indices, val_indices, test_indices = resolve_splits(
        corpus,
        seed=42,
    )
    print(
        f"  Split sizes: train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}"
    )

    source_tokenizer = build_source_tokenizer(corpus, train_indices, encoder_variant)
    target_tokenizer = AmplitudeTokenizer(
        expression_mode=decoder_representation
    ).build_vocab(
        [corpus.targets[index].sequence_target_text() for index in train_indices]
    )
    grammar = RPNGrammar(target_tokenizer) if use_grammar else None
    max_target_len = max(target_tokenizer.max_postfix_tokens + 8, 128)
    print(
        f"  Source vocab={source_tokenizer.vocab_size} | "
        f"Target vocab={target_tokenizer.vocab_size}"
    )

    dataset = QEDAmp2SqDataset(
        corpus=corpus,
        indices=train_indices[:batch_size],
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_src_term_len=192,
        max_src_terms=32,
        max_flat_src_len=3072,
        max_tgt_seq_len=max_target_len,
        encoder_variant=encoder_variant,
    )
    batch = collate_examples([dataset[index] for index in range(len(dataset))])

    model = QEDAmp2SqModel(
        vocab_size=target_tokenizer.vocab_size,
        node_in_dim=0,
        src_vocab_size=source_tokenizer.vocab_size,
        src_pad_id=source_tokenizer.pad_id,
        hidden_dim=64,
        stream_dim=48,
        graph_dim=64,
        num_mp_layers=2,
        num_heads=2,
        dec_d_model=64,
        dec_nhead=2,
        dec_layers=2,
        dec_dim_ff=128,
        max_tgt_seq_len=max_target_len,
        max_src_term_len=192,
        max_flat_src_len=3072,
        pad_id=target_tokenizer.pad_id,
        encoder_variant=encoder_variant,
        target_variant=target_variant,
        use_grammar=use_grammar,
    )
    print(
        f"  Model built with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params"
    )

    tokens = batch.tgt_tokens
    tokens_in = tokens[:, :-1]
    with torch.no_grad():
        outputs = model(batch, tgt_tokens=tokens_in)
        generated = model.generate(
            batch,
            max_len=max_target_len,
            sos_id=target_tokenizer.sos_id,
            eos_id=target_tokenizer.eos_id,
            grammar=grammar,
        )

    print(f"  Sequence logits: {tuple(outputs.sequence_logits.shape)}")
    print(f"  Generated sequence ids: {tuple(generated.sequence_ids.shape)}")
    print(
        f"  Sample generated {decoder_representation}:",
        target_tokenizer.decode(generated.sequence_ids[0].tolist()),
    )
    print("=" * 60)
    print("  Smoke test passed")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="QED canonical amplitude -> squared-amplitude model",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs/custom_qed_amp2sq")
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
            batch_size=args.batch_size or 2,
            target_variant=args.target_variant,
        )
        return

    from .config import QEDAmp2SqConfig
    from .train import QEDAmp2SqTrainer

    config = QEDAmp2SqConfig()
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

    trainer = QEDAmp2SqTrainer(config)
    history = trainer.train()
    metrics = trainer.evaluate()
    payload = {**history, **metrics}
    with open(f"{config.output_dir}/run_summary.json", "w") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
