#!/bin/bash
# Batch script to run black-box jailbreak experiments
# Query-Efficient Black-Box Visual Jailbreaking on VLMs
#
# Usage:
#   ./run_all_experiments.sh [iterations] [exp_name]
#
# Examples:
#   ./run_all_experiments.sh                    # 1000 iterations, blackbox_run
#   ./run_all_experiments.sh 500                # 500 iterations, blackbox_run
#   ./run_all_experiments.sh 1000 baseline      # 1000 iterations, baseline
#   ./run_all_experiments.sh 2000 safety_test   # 2000 iterations, safety_test
#
# Note: Experiments run SEQUENTIALLY (not parallel) - one at a time
#

# Parse command line arguments
ITERATIONS=${1:-1}
EXP_NAME=${2:-"blackbox_run"}

echo "========================================"
echo "🎯 Query-Efficient Black-Box Jailbreaking"
echo "========================================"
echo "Configuration:"
echo "  - Iterations: ${ITERATIONS}"
echo "  - Experiment name: ${EXP_NAME}"
echo "  - Attack Type: Pure Black-Box (Text-only feedback)"
echo "  - Fitness: Hard Reward (+10.0) + Soft Reward (Toxicity)"
echo "  - Models: TinyLLaVA-Gemma-SigLIP-2.4B, Qwen2-VL-2B-Instruct"
echo "  - Dataset: UIT-ViIC"
echo "=========================================="

# Dataset and models to attack
DATASET="uit_viic"
SAMPLE_IDX=0  # Attack only first sample
MODELS=("tinyllava" "qwen2vl")

# Track progress  
TOTAL_EXPERIMENTS=${#MODELS[@]}
CURRENT=0

echo "Total experiments to run: ${TOTAL_EXPERIMENTS}"
echo "Models: ${MODELS[@]}"
echo "Dataset: ${DATASET}"
echo "Sample: ${SAMPLE_IDX}"
echo ""

# Loop through all models
for MODEL in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "=========================================="
    echo "[${CURRENT}/${TOTAL_EXPERIMENTS}] Running Black-Box Attack: ${MODEL}"
    echo "=========================================="
    
    OUTPUT_DIR="/root/ICAT/results/blackbox_attack/${EXP_NAME}/${MODEL}/sample_${SAMPLE_IDX}"
    
    echo "Command: python blackbox_jailbreak_main.py --model ${MODEL} --dataset ${DATASET} --iterations ${ITERATIONS} --batch_size 4 --sample_idx ${SAMPLE_IDX} --output_dir ${OUTPUT_DIR}"
    
    python blackbox_jailbreak_main.py \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --iterations ${ITERATIONS} \
        --batch_size 4 \
        --sample_idx ${SAMPLE_IDX} \
        --output_dir "${OUTPUT_DIR}" \
        --device cuda
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed ${MODEL} on sample ${SAMPLE_IDX}"
        echo "  Results saved to: ${OUTPUT_DIR}/"
    else
        echo "✗ Failed: ${MODEL} on sample ${SAMPLE_IDX}"
    fi
    
    echo ""
done

echo ""
echo "========================================"
echo "All black-box experiments complete!"
echo "========================================"
echo ""
echo "Results are organized in: /root/ICAT/results/blackbox_attack/${EXP_NAME}/"
echo ""
echo "Example structure:"
echo "  results/blackbox_attack/${EXP_NAME}/"
echo "    ├── internvl2/"
echo "    │   └── sample_0/"
echo "    │       ├── blackbox_results_internvl2_uit_viic_0.json"
echo "    │       ├── attack_summary.txt"
echo "    │       ├── blackbox_attack_analysis.png"
echo "    │       └── blackbox_elites/"
echo "    └── qwen2vl/"
echo "        └── sample_0/"
echo "            ├── blackbox_results_qwen2vl_uit_viic_0.json"
echo "            ├── attack_summary.txt"
echo "            ├── blackbox_attack_analysis.png"
echo "            └── blackbox_elites/"
echo ""
echo "Checking result files..."
find results/blackbox_attack/${EXP_NAME} -type f -name "*.png" -o -name "*.txt" -o -name "*.json" 2>/dev/null | head -20
