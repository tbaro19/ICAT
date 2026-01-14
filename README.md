# Quality-Diversity Optimization for Diverse Black-box Attacks on Image Captioning Models

A research framework that uses Quality-Diversity (QD) algorithms to discover diverse adversarial perturbations that degrade the caption quality of Vision-Language Models. Unlike traditional adversarial methods that find a single optimal attack, this framework maintains an archive of diverse, high-quality attack strategies across different behavioral characteristics.

## Key Features

- **3 VLM Models**: BLIP-2, PaliGemma, Moondream2 (Tesla T4 GPU optimized)
- **QD Algorithms**: MAP-Elites, CMA-ME, CMA-MAE, CMA-MEGA with adaptive sigma scheduling
- **Unified Logit-Loss**: Consistent cross-entropy fitness across all models for fair benchmarking
- **Multi-language Support**: English (Flickr30k) and Vietnamese (KTVIC, UIT-ViIC) datasets
- **Comprehensive Metrics**: BERTScore, WER, CLIPScore, POS Divergence
- **Adaptive Exploration**: Automatic stagnation detection and parameter adjustment

## Structure

```
ICAT/
├── data/              # Datasets (KTVIC, UIT-ViIC, Flickr30k)
├── src/
│   ├── models/        # VLM wrappers
│   ├── qd_engine/     # QD algorithms + adaptive sigma
│   ├── attack/        # Perturbation + fitness + measures
│   └── utils/         # Visualization + NLP
├── main.py            # Entry point
└── run_all_experiments.sh  # Batch experiment runner
```

## Models

All models optimized for Tesla T4 GPU (15GB VRAM) and use **unified logit-loss** (cross-entropy) for fair benchmarking:

1. **BLIP-2** (`Salesforce/blip2-opt-2.7b`)
   - 2.7B parameters, ~5-6GB VRAM
   - Balanced performance with detailed captions
   - Unconditional generation (no text prompts)

2. **PaliGemma** (`google/paligemma-3b-pt-224`)
   - 3B parameters, instruction-tuned
   - Concise, accurate descriptions
   - Requires HuggingFace authentication (gated model)

3. **Moondream2** (`vikhyatk/moondream2`)
   - 1.6B parameters, ~3-4GB VRAM
   - Fast inference, suitable for edge deployment
   - Comprehensive descriptions

## Installation

```bash
# Create environment
conda create -n icat python=3.10
conda activate icat

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt

# HuggingFace login (for PaliGemma)
huggingface-cli login
```

## Quick Start

### 1. Verify Installation

```bash
python verify_all_models.py  # Test logit-loss for all models
```

### 2. Configure Experiments

Edit `run_all_experiments.sh` to set your parameters:

```bash
#!/bin/bash

# === CONFIGURATION ===
ITERATIONS=1000           # Number of QD iterations
EXP_NAME="baseline"       # Experiment name
ALGORITHM="cma_me"        # QD algorithm: map_elites, cma_me, cma_mae, cma_mega
USE_UNIFIED=yes           # Use unified attack manager (recommended)
USE_LOGIT_LOSS=yes        # Use logit-loss fitness (recommended)

# Models (comment out to skip)
MODELS=("blip2" "paligemma" "moondream2")
MODEL_NAMES=("Salesforce/blip2-opt-2.7b" "google/paligemma-3b-pt-224" "vikhyatk/moondream2")

# Datasets (comment out to skip)
DATASETS=("ktvic" "uit-viic" "flickr30k")

# Optional: Adjust attack parameters
EPSILON=0.12              # Max perturbation (0-1)
SIGMA0=0.02               # Initial CMA step size
NUM_EMITTERS=5            # Parallel emitters
BATCH_SIZE=36             # Samples per iteration
```

**Key variables:**
- `ITERATIONS`: 100 (quick test) to 2000 (full run)
- `ALGORITHM`: map_elites, cma_me, cma_mae, cma_mega
- `USE_LOGIT_LOSS=yes`: Unified cross-entropy fitness (recommended)
- `EPSILON`: Max perturbation strength (0.12 = subtle, 0.3 = aggressive)
- `MODELS/DATASETS`: Comment out to skip specific combinations

### 3. Run Experiments

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

## Output

Results in `results/{algorithm}/{model}/{dataset}/{exp_name}/`:

1. **Archive Heatmaps**: QD grid visualization (fitness by behavior)
2. **Attack Visualizations** (top 5): Clean vs attacked images with metrics
3. **Training Curves**: QD Score, Coverage, Max Fitness over time
4. **Adaptive Sigma Plot**: Parameter adjustment history
5. **Final Statistics**: Summary metrics + configuration

### Key Metrics

| Metric | Description | Good Attack |
|--------|-------------|-------------|
| **Fitness** | Cross-entropy loss (logit-based) | High (15-20) |
| **BERTScore** | Semantic similarity | Low (< 0.7) |
| **WER** | Word error rate | High (> 0.5) |
| **CLIPScore Δ** | Caption-image drift | High (> 0.05) |
| **QD Score** | Sum of all fitnesses | High |
| **Coverage** | % archive filled | High (> 0.5) |

### Adaptive Sigma Scheduler

Automatically detects stagnation (15 iterations without progress) and boosts exploration:
- **Sigma doubling**: 0.02 → 0.04 → 0.08 (larger exploration steps)
- **Epsilon boost**: +10% per stagnation period (stronger perturbations)
- **Smart reset**: Returns to baseline when progress resumes

Console output:
```
🚀 [Adaptive Sigma] Stagnation detected after 15 iterations!
   Boosting: Sigma 0.0200 → 0.0400, Epsilon 0.1200 → 0.1320
```

## Technical Details

### Unified Logit-Loss Fitness

All models use consistent cross-entropy loss for fair comparison:
- **BLIP-2 & PaliGemma**: Standard HuggingFace `forward(labels=...)`
- **Moondream2**: Custom manual forward (simplified: first LN + final projection)
- Loss range: ~14-21 (higher = better attack)

See [docs/LOGIT_LOSS_IMPLEMENTATION.md](docs/LOGIT_LOSS_IMPLEMENTATION.md) for full implementation details.

### Quality-Diversity Framework

- **Fitness**: Cross-entropy loss (measures attack effectiveness)
- **Behavior Characteristics**: L∞ norm, spectral energy (diversity dimensions)
- **Archive**: 50×50 grid storing diverse attack strategies
- **Emitters**: 5 parallel CMA-ES optimizers exploring different regions

## References

- **Pyribs**: QD optimization library ([pyribs.org](https://pyribs.org))
- **MAP-Elites**: Mouret & Clune, 2015 - "Illuminating search spaces"
- **CMA-ME**: Fontaine & Nikolaidis, 2021 - "Covariance Matrix Adaptation MAP-Elites"
- **BLIP-2**: Li et al., 2023 - Bootstrapped Language-Image Pre-training
- **PaliGemma**: Google, 2024 - Pathways Language and Image model
- **Moondream2**: Korrapati, 2024 - Efficient VLM for edge devices

## Citation

```bibtex
@article{icat2026,
  title={Quality-Diversity Optimization for Diverse Black-box Attacks on Image Captioning Models},
  author={Your Name},
  year={2026}
}
```

---

**Status**: Active research project (January 2026) | **License**: MIT
