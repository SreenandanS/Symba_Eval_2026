# GSoC ML4SCI SYMBA Evaluation Tasks

## Project

**Physics-Informed Encoding and Decoding for Squared Amplitude Calculation**

This repository contains my evaluation work for the ML4SCI SYMBA project. The work here focuses on two parts of the evaluation:

- **Common Task 1.2: Dataset preprocessing**
- **Specific Task 2.1: Train / Evaluate advanced model**

The generated `outputs/` directory is too large. 
Experiment outputs, summaries, and other large artifacts are available on Google Drive:

- [SYMBA evaluation outputs](https://drive.google.com/drive/folders/1BUmaO2sva8YHfezcYzSHoBdY3oQPRGaa?usp=share_link)

## Common Task 1.2: Dataset Preprocessing

The first part of the work is to make the symbolic dataset usable for learning. Each SYMBA line is split into interaction text, topology text, raw amplitude, and raw squared amplitude. From there, the source is normalized with bounded local placeholders so that sample-specific labels become reusable tokens instead of exploding the vocabulary.

Examples:

- `%gam_5702 -> %gam_INDEX_1`
- `p_3 -> MOMENTUM_0`
- `1/6` stays `1/6`

The squared target is also rewritten into a cleaner factorized form with a prefactor, numerator, and denominator. This makes the decoder target much shorter and more structured.

![Feynman diagram, amplitude, and squared amplitude](images/feynman_diagram_amp_sq.png)

![Factorized target vs raw target](images/factorization_image.png)

Useful Task 1 numbers:

- dataset size: QCD `234`, QED `360`
- raw amplitude vocab: QCD `6892`, QED `3449`
- bounded amplitude vocab: QCD `694`, QED `198`
- raw squared postfix vocab: QCD `84`, QED `39`
- factorized postfix vocab: QCD `29`, QED `32`
- average bounded source length: QCD `429.72`, QED `120.87`
- average raw squared postfix length: QCD `326.49`, QED `69.52`
- average factorized postfix length: QCD `46.64`, QED `42.70`

More detail: [Common Task 1.2/README.md](Common%20Task%201.2/README.md)

## Specific Task 2.1: Train / Evaluate Advanced Model

The second part of the work is to train and evaluate advanced models that use the structure of the source instead of treating everything as flat text.

There are four main pipelines in the repository:

- `custom_qcd_fd2sq`: QCD diagram -> squared amplitude
- `custom_qed_fd2sq`: QED diagram -> squared amplitude
- `custom_qcd_amp2sq`: QCD amplitude -> squared amplitude
- `custom_qed_amp2sq`: QED amplitude -> squared amplitude

All four pipelines share the same target-side idea. Instead of decoding only the raw SYMBA squared-amplitude text, I also study a factorized target where the squared amplitude is rewritten as `color_factor * numerator / denominator` for QCD or `charge_factor * numerator / denominator` for QED. The decoder then generates postfix symbolic sequences under grammar constraints, so the comparison between models is mainly about how well the encoder captures the underlying physics structure.

### QCD fd2sq

For QCD diagram to squared-amplitude prediction, the encoder treats each tree-level `2 -> 2` diagram as a 7-node graph with `4` external particles, `2` vertices, and `1` propagator. On top of this graph, it runs three separate physics-aligned message-passing streams:

- a Lorentz / kinematic stream for momentum signatures, mass features, and `s/t/u` channel labels
- an SU(3) color-flow stream for fundamental, anti-fundamental, and adjoint color structure
- a spinor / fermion-line stream for fermion number, line identity, and interaction type

After each layer, the streams exchange information through cross-stream exchange, and the final node states are fused into a unified graph memory. The model then builds global color, denominator, and spinor context tokens before decoding.

For the diagram side, the models use structured encoders matched to the interaction:

![QCD fd2sq architecture](images/qcd_fd2sq_architecture.png)

### QED fd2sq

For QED, the interaction topology is much more rigid, so I use a fixed 7-slot interaction contract instead of a fully flexible graph. The slots correspond to `4` external particles, `2` vertices, and `1` propagator, each with particle-type, flavor, charge, mass, and direction features. Message passing happens over two physically motivated relation systems:

- channel / propagator relations connecting external legs, vertices, and the propagator
- fermion-line relations following directed fermion flow

These relation streams are fused with the base slot stream, and the encoder then constructs global context tokens such as topology, charge, and denominator summaries for the decoder.

![QED fd2sq architecture](images/qed_fd2sq_architecture.png)

### QCD and QED amp2sq

For amplitude to squared-amplitude prediction, the source is not a diagram graph but a dense symbolic expression. Here I use a physics-tagged structured sequence encoder. Each raw amplitude is first canonicalized with bounded normalization of indices and momenta, and then rewritten into one tagged source sequence with three sections:

- `[RAW_SRC]` for bounded raw amplitude tokens
- `[GLOBAL]` for global metadata such as coupling powers, rational prefactors, denominator tokens, and theory-specific labels
- `[TERM_SUMMARIES]` for per-term physics features such as coefficient pattern, atom counts, flavor counts, chain lengths, and momentum slots

A Transformer encoder processes this enriched sequence. Section-specific pools are then fused into condition tokens prepended to the source memory, so the decoder can attend both to token-level detail and to compact physics summaries.

For the amplitude side, the models use physics-tagged structured source sequences:

![QCD and QED amp2sq architecture](images/qcd_qed_amp2sq_arch.png)

The main evaluation summaries are shown below.

### Sequence Accuracy

![Sequence accuracy summary](images/ablation_study_all.png)

### Parameter Comparison

![Parameter comparison](images/parameters_all.png)

### Sequence Length and Epochs

![Sequence length and epochs](images/seqlen_epochs.png)

More detail: [Specific Task 2.1/README.md](Specific%20Task%202.1/README.md)

## Repository Structure

```text
.
├── dataset/                         tree-level QCD and QED SYMBA text files
├── images/                          figures used in the evaluation write-up
├── Common Task 1.2/                 dataset preprocessing materials
│   ├── tokenization-eda-2.ipynb
│   └── README.md
├── Specific Task 2.1/               train / evaluate advanced model materials
│   ├── custom_qcd_fd2sq/
│   ├── custom_qcd_amp2sq/
│   ├── custom_qed_fd2sq/
│   ├── custom_qed_amp2sq/
│   ├── qcd-fdtosq.ipynb
│   ├── qcd-amptosq.ipynb
│   ├── qed-fdtosq.ipynb
│   ├── qed-amptosq.ipynb
│   └── README.md
```
## Thank you!
Please mail me at sreenandan.shashidharan@gmail.com or at 24JE0701@iitism.ac.in if anything is amiss. I sincerely apologise in advance. 
