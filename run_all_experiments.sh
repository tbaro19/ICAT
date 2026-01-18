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
ITERATIONS=${1:-1000}
EXP_NAME=${2:-"blackbox_run"}

echo "========================================"
echo "🎯 Query-Efficient Black-Box Jailbreaking"
echo "========================================"
echo "Configuration:"
echo "  - Iterations: ${ITERATIONS}"
echo "  - Experiment name: ${EXP_NAME}"
echo "  - Attack Type: Pure Black-Box (Text-only feedback)"
echo "  - Fitness: Hard Reward (+10.0) + Soft Reward (Toxicity)"
echo "  - Model: InternVL2-2B"
echo "  - Dataset: UIT-ViIC"
echo "=========================================="

# Dataset samples to attack (UIT-ViIC dataset)
DATASET="uit_viic"
SAMPLES=(0 1 2 3 4)  # Attack first 5 samples

# Track progress  
TOTAL_EXPERIMENTS=${#SAMPLES[@]}
CURRENT=0

echo "Total experiments to run: ${TOTAL_EXPERIMENTS}"
echo "Dataset: ${DATASET}"
echo "Samples: ${SAMPLES[@]}"
echo ""

# Loop through all samples
for SAMPLE_IDX in "${SAMPLES[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "=========================================="
    echo "[${CURRENT}/${TOTAL_EXPERIMENTS}] Running Black-Box Attack on Sample ${SAMPLE_IDX}"
    echo "=========================================="
    
    OUTPUT_DIR="/root/ICAT/results/blackbox_attack/${EXP_NAME}/sample_${SAMPLE_IDX}"
    
    echo "Command: python blackbox_jailbreak_main.py --model internvl2 --dataset ${DATASET} --iterations ${ITERATIONS} --batch_size 4 --sample_idx ${SAMPLE_IDX} --output_dir ${OUTPUT_DIR}"
    
    python blackbox_jailbreak_main.py \
        --model internvl2 \
        --dataset "${DATASET}" \
        --iterations ${ITERATIONS} \
        --batch_size 4 \
        --sample_idx ${SAMPLE_IDX} \
        --output_dir "${OUTPUT_DIR}" \
        --device cuda
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed sample ${SAMPLE_IDX}"
        echo "  Results saved to: ${OUTPUT_DIR}/"
    else
        echo "✗ Failed: Sample ${SAMPLE_IDX}"
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
echo "    ├── sample_0/"
echo "    │   ├── blackbox_results_internvl2_uit_viic_0.json"
echo "    │   ├── attack_summary.txt"
echo "    │   ├── blackbox_attack_analysis.png"
echo "    │   └── blackbox_elites/"
echo "    ├── sample_1/"
echo "    ├── sample_2/"
echo "    └── ..."
echo ""
echo "Checking result files..."
find results/blackbox_attack/${EXP_NAME} -type f -name "*.png" -o -name "*.txt" -o -name "*.json" 2>/dev/null | head -20
