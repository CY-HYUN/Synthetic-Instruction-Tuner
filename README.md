# Synthetic Instruction Tuner

> Zero-cost LLM fine-tuning pipeline with synthetic data generation, SFT, and DPO alignment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/)

## Overview

**NLP Task**: Instruction-Following Text Generation for Multi-Domain Dialogue Systems

This project develops an instruction-following LLM capable of generating high-quality responses across various domains (coding, reasoning, creative writing, etc.) through a complete zero-cost synthetic data pipeline:

1. **Magpie Data Generation** - 1.5K synthetic instruction-response pairs
2. **Quality Filtering** - Rule-based filtering to 1K high-quality samples
3. **Preference Generation** - 600 preference pairs with reward model
4. **SFT Training** - Supervised fine-tuning with LoRA
5. **DPO Training** - Direct preference optimization
6. **Evaluation** - Benchmarks + agent capability tests

**Cost**: $0 (runs on free Google Colab T4 GPU)

## Quick Start

### Prerequisites
- Google account for Colab
- Hugging Face account with Llama access

### Run Notebooks Sequentially

| Notebook | Description | Time |
|----------|-------------|------|
| `01_setup.ipynb` | Environment setup | 10 min |
| `02_magpie_generation.ipynb` | Generate synthetic data | 16-17 hrs |
| `03_quality_filtering.ipynb` | Filter data | 30 min |
| `04_preference_generation.ipynb` | Create preference pairs | 6-8 hrs |
| `05_sft_training.ipynb` | SFT training | 6-8 hrs |
| `06_dpo_training.ipynb` | DPO training | 4-6 hrs |
| `07_benchmark_evaluation.ipynb` | Evaluate models | 3-4 hrs |
| `08_agent_evaluation.ipynb` | Agent benchmarks | 2-3 hrs |

## Project Structure

```
synthetic-instruction-tuner/
├── notebooks/           # 8 Colab notebooks (full pipeline)
├── src/                 # Python modules
│   ├── filtering/       # Quality filter
│   └── preference/      # Preference generator
├── data/                # Generated datasets (gitignored)
├── models/              # Model checkpoints (gitignored)
└── evaluation/          # Results & figures (gitignored)
```

## Technical Details

### Models
- **Data Generation**: Llama-3.1-8B-Instruct
- **Fine-tuning**: Llama-3.2-3B / Mistral-7B / Qwen2.5-3B
- **Reward Model**: OpenAssistant DeBERTa-v3

### Configuration
- **LoRA**: r=8, alpha=16, 4-bit quantization
- **SFT**: 3 epochs, lr=2e-4, batch_size=4
- **DPO**: 1 epoch, beta=0.1, lr=5e-5

### Data Pipeline
```
1.5K raw samples → 1K filtered → 600 preference pairs → Fine-tuned model
```

## Results

Expected improvements after SFT+DPO:
- **Instruction Following**: +10-15% accuracy
- **Response Quality**: Better structure and coherence
- **Agent Capabilities**: Improved multi-turn conversations

## References

- [Magpie Paper](https://arxiv.org/abs/2406.08464)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## License

MIT License - see [LICENSE](LICENSE) for details.

---
