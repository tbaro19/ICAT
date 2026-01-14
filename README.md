# Attacking Vision-Language Models with Quality-Diversity Algorithms

Adversarial attack framework for Vision-Language Models (BLIP-2, PaliGemma, Moondream2) using Quality-Diversity optimization to find diverse perturbations that degrade caption quality.

## Key Features

- **3 VLM Models**: BLIP-2, PaliGemma, Moondream2 (Tesla T4 GPU optimized)
- **QD Algorithms**: MAP-Elites, CMA-ME, CMA-MAE, CMA-MEGA with adaptive sigma scheduling
- **Unified Logit-Loss**: Consistent cross-entropy fitness across all models for fair benchmarking
- **Multi-language**: English (Flickr30k) and Vietnamese (KTVIC, UIT-ViIC)
- **Comprehensive Metrics**: BERTScore, WER, CLIPScore, POS Divergence

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

### Supported Vision-Language Models

All models are optimized for Tesla T4 GPU (15GB VRAM):

1. **BLIP-2** (`Salesforce/blip2-opt-2.7b`)
   - Parameters: 2.7B
   - Memory: ~5-6GB VRAM
   - Caption Generation: Unconditional generation (no text prompts)
   - Optimizations: max_new_tokens=150, min_length=40, length_penalty=1.5, repetition_penalty=1.2
   - Best for: Balanced performance, detailed captionsanguage model
   - Best for: Balanced performance

2. **PaliGemma** (`google/paligemma-3b-pt-224`)
   - Parameters: 3B
   - Caption Generation: Simple "describe" prompt with prefix removal
   - Note: Requires HuggingFace authentication (gated model)
   - Best for: Instruction-tuned understanding, concise descriptions
1. **BLIP-2** (2.7B): Balanced performance, detailed captions
2. **PaliGemma** (3B): Instruction-tuned, requires HF auth
3. **Moondream2** (1.6B): Fast inference, edge deployment

All use **unified logit-loss** (cross-entropy) for fair benchmarking.

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Optional: For Vietnamese NLP
pip install underthesea pyvi
```

### 3. Download Datasets

```bash
# KTVIC - Vietnamese Life Domain
```bash
# Environment
conda create -n icat python=3.10
conda activate icat

# PyTorch + CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Dependencies
pip install -r requirements.txt

# HuggingFace login (for PaliGemma)
huggingface-cli login
```
    --iterations 500 \
    --exp_name test_run
```

### Common Configurations

#### Fast Testing (2 iterations)
```bash
python main.py --model moondream2 --model_name vikhyatk/moondream2 \
    --dataset ktvic --algorithm cma_me --iterations 2 --exp_name quick_test
```

#### Full Experiment (500 iterations)
```bash
python main.py --model blip2 --model_name Salesforce/blip2-opt-2.7b \
    --dataset flickr30k --algorithm cma_me --iterations 500 --exp_name full_run
```

#### PaliGemma with Authentication
```bash
python main.py --model paligemma --model_name google/paligemma-3b-pt-224 \
    --dataset uit-viic --algorithm cma_me --iterations 500 --exp_name paligemma_run
```

## Comprehensive Experiments

### Run All Models on All Datasets

Create a batch script:

```bash
cat > run_all_experiments.sh << 'EOF'
#!/bin/bash

# Models to test
MODELS=("blip2" "paligemma" "moondream2")
MODEL_NAMES=("Salesforce/blip2-opt-2.7b" "google/paligemma-3b-pt-224" "vikhyatk/moondream2")

# Datasets
DATASETS=("ktvic" "uit-viic" "flickr30k")

# Run all combinations
for i in {0..2}; do
    model="${MO
```bash
python verify_all_models.py  # Verify logit-loss for all models
```

### Run Experiments

Edit `run_all_experiments.sh` to configure:

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

# === RUN ===
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

**Key variables to adjust:**
- `ITERATIONS`: 100 (quick test) to 2000 (full run)
- `ALGORITHM`: Try different QD algorithms
- `USE_LOGIT_LOSS=yes`: Enables unified cross-entropy fitness
- `EPSILON`: Higher = stronger attacks (but less subtle)
- Models/Datasets: Comment out lines to skip specific combinations
3. **BERTScore (Semantic Similarity)**
   - **What it measures**: Semantic similarity between clean and attacked captions
   - **Range**: [0, 1], lower = more semantic degradation
   - **Method**: Uses BERT embeddings with token-level matching
   - **Components**:
     - Precision: Attacked caption tokens covered by clean caption
     - Recall: Clean caption tokens covered by attacked caption
     - F1: Harmonic mean (primary metric shown)
   - **Interpretation**: F1 < 0.7 indicates significant semantic change
   - **Library**: bert-score (multilingual support)

7. **QD Score (Quality-Diversity Score)**
   - Sum of fitness across all elites in archive
   - Balances attack effectiveness with behavioral diversity
   - Formula: `Σ fitness(elite) for all elites`
   - Higher = better overall performance

8. **Coverage (Archive Fill Ratio)**
   - Percentage of grid cells with solutions
   - Range: [0, 1], e.g., 0.30 = 30% of cells filled
   - Formula: `occupied_cells / total_cells`
   - Higher = more diverse attack strategies
   - Example: 43/144 = 30% coverage (stagnation point for BLIP-2)

### Metric Interpretation Summary

| Metric | Good Attack | Poor Attack | Notes |
|--------|-------------|-------------|-------|
| Fitness | High (0.5-0.8) | Low (< 0.2) | Primary objective |
| BERTScore F1 | Low (< 0.7) | High (> 0.9) | Lower = more semantic change |
| WER | High (> 0.5) | Low (< 0.2) | Higher = more words changed |
| CLIPScore Δ | High (> 0.05) | Low (< 0.01) | Caption drifted from image |
| POS Divergence | High (> 0.3) | Low (< 0.1) | Different objects described |
| QD Score | High | Low | Sum of all fitness values |
| Coverage | High (> 0.5) | Low (< 0.2) | More cells filled |
     - Insertions: Words added
     - Deletions: Words removed
   - **Formula**: `WER = (S + I + D) / N` where N = words in reference
   - **Interpretation**: WER > 0.5 means over half the words changed
   - **Library**: jiwer

5. **CLIPScore (Multimodal Image-Text Alignment)**
   - **What it measures**: How well text describes the image
   - **Range**: [-1, 1], typically [0.2, 0.4]
   - **Method**: Cosine similarity between CLIP image and text embeddings
   - **Two measurements**:
     - Clean: Original image ↔ clean caption (baseline)
     - Attacked: Original image ↔ attacked caption
   - **Delta (Δ)**: Clean - Attacked (positive = caption drifted from image)
   - **Interpretation**: Δ > 0.05 shows significant caption drift
   - **Note**: Uses VLM's internal CLIP-based embeddings

6. **POS Divergence (Syntactic Structure)**
   - **What it measures**: Changes in grammatical structure
   - **Range**: [0, 1], higher = more structural change
   - **Method**: Part-of-speech tag counting
   Adaptive Sigma Scheduler

### Overview
The **Adaptive Sigma Scheduler** automatically detects stagnation and adjusts exploration parameters to overcome optimization plateaus. This is especially important for resilient models like BLIP-2 where the Q-Former creates a challenging optimization landscape.

### How It Works

1. **Stagnation Detection**
   - Monitors number of elites in archive every iteration
   - Triggers when elite count doesn't increase for 15 consecutive iterations
   - Example: Stuck at 43/144 elites for 15 iterations

2. **Automatic Parameter Boost**
   - **Sigma (σ)**: Doubles evolution step size (0.02 → 0.04 → 0.08...)
     - Forces emitters to make long-range jumps
     - Helps escape local optima
   - **Epsilon (ε)**: Increases by 10% (0.12 → 0.132)
     - Provides physical perturbation boost
     - Allows exploration beyond initial bounds

3. **Smart Reset**
   - When progress detected (new elite added):
     - Immediately resets sigma to baseline (0.02)
     - Reverts epsilon to original (0.12)
     - Resumes fine-grained exploitation
   - Prevents "monochromatic" heatmaps
   Output

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

### Adaptive Sigma

Automatically detects stagnation (15 iterations without progress) and boosts exploration:
- Doubles sigma (0.02 → 0.04)
- Increases epsilon by 10%
- Resets when progress resumes

Watch console for: `🚀 [Adaptive Sigma] Stagnation detected!`References

- **Pyribs**: QD optimization (pyribs.org)
- **BLIP-2**: Li et al., 2023 (Salesforce LAVIS)
- **PaliGemma**: Google, 2024
- **Moondream2**: Korrapati, 2024
- **Unified Logit-Loss**: See [docs/LOGIT_LOSS_IMPLEMENTATION.md](docs/LOGIT_LOSS_IMPLEMENTATION.md)

---

**Status**: Active research projec