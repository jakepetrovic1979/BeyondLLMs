# PNA Research Paper

This directory contains the LaTeX source for the Phenomenal Nested Architectures research paper.

## Files

- **pna_paper.tex** - Main LaTeX document
- **figures/** - Directory containing all figures referenced in the paper
  - factorial_results.png - Factorial analysis visualization
  - budget_curve.png - Replay buffer efficiency curve

## Compiling the Paper

### Requirements

You need a LaTeX distribution installed:

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install texlive-full
```

**macOS:**
```bash
brew install --cask mactex
```

**Windows:**
Download and install [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)

### Compilation Commands

#### Using pdflatex (recommended):

```bash
cd paper
pdflatex pna_paper.tex
bibtex pna_paper
pdflatex pna_paper.tex
pdflatex pna_paper.tex
```

The multiple runs are necessary to resolve references and citations.

#### Using latexmk (automated):

```bash
cd paper
latexmk -pdf pna_paper.tex
```

#### Clean auxiliary files:

```bash
cd paper
latexmk -c  # Clean most auxiliary files
latexmk -C  # Clean all generated files including PDF
```

### Output

The compiled PDF will be named `pna_paper.pdf`

## Paper Structure

1. **Abstract** - Overview of PNA framework and key results
2. **Introduction** - Motivation and transformer limitations
3. **Theoretical Foundations** - MPE, IIT, GWT, and nested learning
4. **Architecture** - Four-level PNA hierarchy
5. **Methodology** - Experimental setup and metrics
6. **Results** - Factorial analysis and continual learning benchmarks
7. **Discussion** - Dual-benefit mechanism and consciousness metrics
8. **Conclusion** - Summary and future directions
9. **Appendices** - Confidence intervals and replay budget analysis

## Key Contributions

- **Consciousness-first AGI framework** treating phenomenal experience as foundational
- **Multi-timescale architecture** addressing catastrophic forgetting
- **Experimental validation** showing >80% retention vs <20% for baselines
- **Consciousness metrics** (Φ, B, Ω, R) exceeding theoretical thresholds
- **25× replay efficiency** compared to standard architectures

## Citation

If you use this work, please cite:

```bibtex
@article{petrovic2025pna,
  title={Phenomenal Nested Architectures: A Consciousness-First Framework for Artificial General Intelligence},
  author={Petrovic, Zeljko},
  journal={arXiv preprint},
  year={2025}
}
```

## Updates and Revisions

- **December 2025** - Initial draft with Split-MNIST experiments
- TODO: Add Continual ImageNet results
- TODO: Add language task benchmarks
- TODO: Full perturbational IIT analysis

## Contact

For questions or collaboration:
- Email: [email protected]
- GitHub: https://github.com/jakepetrovic1979/BeyondLLMs
