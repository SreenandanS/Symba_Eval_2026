# QCD Feynman Diagram -> Squared Amplitude

Three-stream symmetry-aware equivariant GNN for predicting squared amplitudes from QCD tree-level 2->2 Feynman diagrams.

## Architecture Snapshot

![QCD fd2sq architecture](../../images/qcd_fd2sq_architecture.png)

## Architecture

The encoder represents each diagram as a fixed 7-node graph (4 external particles, 2 vertices, 1 propagator) and performs relation-aware message passing over three physics-aligned streams:

- **Lorentz / Kinematic** (`lorentz_mp.py`) — momentum signatures, mass features, s/t/u channel labels
- **SU(3) Color-Flow** (`color_mp.py`) — fundamental / anti-fundamental / adjoint color representations
- **Spinor / Fermion-Line** (`spinor_mp.py`) — fermion number, line identity, vertex interaction type

Streams exchange information via `CrossStreamExchange` after each layer and fuse via gated `CrossStreamAttention`. The decoder is an autoregressive Transformer over postfix symbolic sequences with `RPNGrammar` constraints.

## Results

| Target | Encoder | Test Seq Accuracy |
| --- | --- | ---: |
| factorized | custom | **1.0000** |
| factorized | seq2seq | 0.7917 |
| raw_string | custom | 0.7083 |
| raw_string | seq2seq | 0.4167 |

234 QCD diagrams, 80/10/10 split, seed 42.

## Package Structure

| File | Role |
| --- | --- |
| `parser.py` | SYMBA text -> `FeynmanDiagram` objects |
| `feynman_graph.py` | Diagram -> PyG `Data` with stream-specific relations |
| `contracts.py` | Node-role and relation constants |
| `lorentz_mp.py` | Kinematic message passing |
| `color_mp.py` | Color-flow message passing |
| `spinor_mp.py` | Spinor / fermion-line message passing |
| `encoder.py` | Three-stream equivariant GNN encoder |
| `baseline_encoder.py` | Serialized-diagram seq2seq baseline encoder |
| `sequence_encoder.py` / `sequence_dataset.py` | Seq2seq data pipeline |
| `factorization.py` | Factorized target construction |
| `tokenizer.py` | Symbolic tokenization (literal-preserving) |
| `grammar.py` | Postfix RPN grammar masks |
| `model.py` | Full encoder-decoder model |
| `dataset.py` | Graph dataset and dataloaders |
| `splits.py` | Train/val/test split logic |
| `train.py` | Training loop |
| `config.py` | Dataclass configuration |
| `run.py` | CLI entrypoint |

## Usage

```bash
# Smoke test
PYTHONPATH="Specific Task 2.1" python -m custom_qcd_fd2sq.run --smoke-test --data-dir dataset --encoder-variant custom

# Full training
PYTHONPATH="Specific Task 2.1" python -m custom_qcd_fd2sq.run --data-dir dataset --output-dir outputs/custom_qcd_fd2sq --encoder-variant custom --target-variant factorized
```

## Previous Approaches Tried

1. **Pure seq2seq over serialized diagram text** — natural starting point; ignores diagram topology and physics structure entirely. Kept as baseline.
2. **Single-stream GNN without physics-specific streams** — early prototype; underperformed the multi-stream design because it conflated kinematic, color, and spinor information.
3. **Three-stream symmetry-aware equivariant GNN** (current) — separate Lorentz, SU(3) color-flow, and spinor message-passing streams with cross-stream exchange. Best result: 100% test accuracy on factorized target.
