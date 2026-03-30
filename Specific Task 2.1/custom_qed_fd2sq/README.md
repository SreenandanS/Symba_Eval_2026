# QED Feynman Diagram -> Squared Amplitude

Dual-relation fixed-slot interaction encoder for predicting squared amplitudes from QED tree-level 2->2 Feynman diagrams.

## Architecture Snapshot

![QED fd2sq architecture](../../images/qed_fd2sq_architecture.png)

## Architecture

The encoder uses a fixed 7-slot interaction contract (4 external particles, 2 vertices, 1 propagator) with per-slot features (flavor, particle type, charge, mass, incoming/outgoing, quark/lepton/photon flags). Message passing operates over two physically motivated relation systems:

- **Channel / Propagator Relations** — external-to-vertex leg edges and vertex-to-propagator edges with s/t/u channel annotations
- **Fermion-Line Relations** — directed edges along/against fermion flow, plus boson edges

After message passing, the encoder constructs a topology state token (from vertex/propagator slots + topology features) and a static charge token. These join the fused slot memories as decoder memory. The decoder is an autoregressive Transformer over postfix symbolic sequences with `RPNGrammar` constraints.

## Results

| Target | Encoder | Test Seq Accuracy |
| --- | --- | ---: |
| factorized | custom | 0.9722 |
| factorized | seq2seq | 0.9722 |
| raw_string | custom | **1.0000** |
| raw_string | seq2seq | **1.0000** |

360 QED diagrams, 80/10/10 split, seed 42.

## Package Structure

| File | Role |
| --- | --- |
| `parser.py` | SYMBA text -> `QEDInteraction` objects |
| `interaction.py` | Canonical QED interaction contract |
| `features.py` | Per-slot and global features |
| `relations.py` | Channel/propagator and fermion-line relation tensors |
| `contracts.py` | Slot-role and relation constants |
| `feynman_graph.py` | Fixed-slot tensor construction |
| `encoder.py` | Dual-relation fixed-slot encoder |
| `sequence_encoder.py` / `sequence_dataset.py` | Seq2seq baseline pipeline |
| `factorization.py` | Factorized target construction |
| `tokenizer.py` | Symbolic tokenization (literal-preserving) |
| `grammar.py` | Postfix RPN grammar masks |
| `model.py` | Full encoder-decoder model |
| `dataset.py` | Dataset and dataloaders |
| `splits.py` | Train/val/test split logic |
| `train.py` | Training loop |
| `config.py` | Dataclass configuration |
| `run.py` | CLI entrypoint |

## Usage

```bash
# Smoke test
PYTHONPATH="Specific Task 2.1" python -m custom_qed_fd2sq.run --smoke-test --data-dir dataset --encoder-variant custom

# Full training
PYTHONPATH="Specific Task 2.1" python -m custom_qed_fd2sq.run --data-dir dataset --output-dir outputs/custom_qed_fd2sq --encoder-variant custom --target-variant factorized
```

## Previous Approaches Tried

1. **Pure seq2seq over serialized interaction text** — natural starting point; ignores interaction topology. Kept as baseline.
2. **PyG-style generic graph encoding** — mirrored the QCD graph pipeline; moved away because the rigid QED tree-level 2->2 topology is better captured by fixed slots than a generic graph.
3. **Dual-relation fixed-slot interaction encoder** (current) — channel/propagator and fermion-line message passing over a 7-slot contract matched to QED's U(1) structure. Both encoders reach parity on this dataset.
