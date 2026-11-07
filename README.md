# Temporal Grounding for Multimodal Dialogue Agents

**Status: code released.**

This repository contains the implementation for the paper:

> **Temporal Grounding for Multimodal Dialogue Agents**
> Maria Hernandez, Wei Zhang, Daniel Okafor, Priya Raman
> EMNLP 2023

The implementation includes the temporal grounding encoder, the dialogue
state tracker and the full training/evaluation pipeline used in the paper.

## Repository layout

- `src/` - model, data pipeline, training and evaluation code
- `configs/` - experiment configuration files
- `scripts/` - launch scripts

## Getting started

```bash
pip install -r requirements.txt
python -m src.train --config configs/default.yaml
```

## Citation

If you use this code, please cite the paper above.
