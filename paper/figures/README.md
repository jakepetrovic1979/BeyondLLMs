# Figures Directory

This directory contains figures referenced in the PNA paper (`pna_paper.tex`).

## Required Figures

The paper currently references the following figures:

1. **factorial_results.png** - Factorial analysis showing PNA vs Replay effects
2. **budget_curve.png** - Replay buffer size efficiency comparison

## Generating Figures

To generate the actual figures from experimental data:

```bash
# From the repository root
python pna/utils/visualization.py --experiment factorial --output paper/figures/factorial_results.png
python pna/utils/visualization.py --experiment budget --output paper/figures/budget_curve.png
```

## Placeholder Figures

Currently, placeholder images are provided. Replace these with actual experimental results before publication.

## Figure Specifications

- Format: PNG (300 DPI recommended for publication)
- Width: Optimized for 0.85\linewidth in LaTeX (approximately 5.5 inches)
- Color scheme: Colorblind-friendly palette
- Labels: Clear, readable at reduced size
