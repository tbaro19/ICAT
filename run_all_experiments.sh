#!/bin/bash
# Batch script to run all model+dataset combinations
# Quality-Diversity Black-box Attacks on Vision-Language Models
#
# Usage:
#   ./run_all_experiments.sh [iterations] [exp_name] [qd_algo] [use_unified] [use_logit_loss]
#
# Examples:
#   ./run_all_experiments.sh                                 # 1000 iterations, main_run, cma_me, no unified
#   ./run_all_experiments.sh 50                              # 50 iterations, main_run, cma_me, no unified
#   ./run_all_experiments.sh 500 baseline                    # 500 iterations, baseline, cma_me, no unified
#   ./run_all_experiments.sh 500 baseline cma_mega           # 500 iterations, baseline, cma_mega, no unified
#   ./run_all_experiments.sh 1000 unified_test cma_me yes    # 1000 iterations, unified framework (similarity)
#   ./run_all_experiments.sh 1000 unified_full cma_me yes yes # 1000 iterations, unified + logit-loss
#
# Note: Experiments run SEQUENTIALLY (not parallel) - one at a time
#

# Parse command line arguments
ITERATIONS=${1:-1000}
EXP_NAME=${2:-"main_run"}
USE_UNIFIED=${3:-"yes"}
USE_LOGIT_LOSS=${4:-"yes"}

# All QD algorithms to run
ALGORITHMS=("map_elites" "cma_me" "cma_mae")

echo "========================================"
echo "Running All Model x Dataset x Algorithm Experiments"
echo "========================================"
echo "Configuration:"
echo "  - Iterations: ${ITERATIONS}"
echo "  - Experiment name: ${EXP_NAME}"
echo "  - QD Algorithms: map_elites, cma_me, cma_mae"
echo "  - Use Unified Framework: ${USE_UNIFIED}"
echo "  - Use Logit-Loss Fitness: ${USE_LOGIT_LOSS}"
echo "=========================================="

# Build unified framework flags
UNIFIED_FLAGS=""
if [ "${USE_UNIFIED}" = "yes" ] || [ "${USE_UNIFIED}" = "y" ] || [ "${USE_UNIFIED}" = "true" ]; then
    UNIFIED_FLAGS="--use_unified"
    echo "🎯 Unified Adaptive Adversarial Framework: ENABLED"
    
    if [ "${USE_LOGIT_LOSS}" = "yes" ] || [ "${USE_LOGIT_LOSS}" = "y" ] || [ "${USE_LOGIT_LOSS}" = "true" ]; then
        UNIFIED_FLAGS="${UNIFIED_FLAGS} --use_logit_loss"
        echo "   - Fitness Mode: Logit-Loss (Cross-Entropy)"
    else
        echo "   - Fitness Mode: Similarity-Based (CLIP)"
    fi
else
    echo "📊 Standard MAP-Elites System: ENABLED"
    echo "   - Fitness Mode: Similarity-Based (CLIP)"
fi
echo "=========================================="

# Model configurations (optimized for Tesla T4)
MODEL_BLIP2="Salesforce/blip2-opt-2.7b"
MODEL_PALIGEMMA="google/paligemma-3b-pt-224"
MODEL_MOONDREAM2="vikhyatk/moondream2"

# Datasets
DATASETS=("uit-viic")

# Track progress
TOTAL_EXPERIMENTS=$((3 * ${#DATASETS[@]} * ${#ALGORITHMS[@]}))
CURRENT=0

echo "Total experiments to run: ${TOTAL_EXPERIMENTS}"
echo "Algorithms: ${ALGORITHMS[@]}"
echo "Models: BLIP-2, PaliGemma, Moondream2"
echo "Datasets: ${DATASETS[@]}"
echo ""

# Loop through all algorithms
for QD_ALGO in "${ALGORITHMS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Starting Algorithm: ${QD_ALGO}"
    echo "=========================================="
    echo ""

# Run BLIP-2 experiments
echo "========== BLIP-2 Experiments =========="
for dataset in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[${CURRENT}/${TOTAL_EXPERIMENTS}] Running BLIP-2 on ${dataset}..."
    echo "Command: python main.py --model blip2 --model_name ${MODEL_BLIP2} --dataset ${dataset} --algorithm ${QD_ALGO} --iterations ${ITERATIONS} --exp_name ${EXP_NAME} ${UNIFIED_FLAGS}"
    
    python main.py \
        --model blip2 \
        --model_name "${MODEL_BLIP2}" \
        --dataset "${dataset}" \
        --algorithm "${QD_ALGO}" \
        --iterations ${ITERATIONS} \
        --exp_name "${EXP_NAME}" \
        ${UNIFIED_FLAGS}
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed BLIP-2 on ${dataset}"
        echo "  Results saved to: results/${QD_ALGO}/blip2_Salesforce_blip2-opt-2.7b/${dataset}/${EXP_NAME}/"
    else
        echo "✗ Failed: BLIP-2 on ${dataset}"
    fi
done

# Run PaliGemma experiments
echo ""
echo "========== PaliGemma Experiments =========="
for dataset in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[${CURRENT}/${TOTAL_EXPERIMENTS}] Running PaliGemma on ${dataset}..."
    echo "Command: python main.py --model paligemma --model_name ${MODEL_PALIGEMMA} --dataset ${dataset} --algorithm ${QD_ALGO} --iterations ${ITERATIONS} --exp_name ${EXP_NAME} ${UNIFIED_FLAGS}"
    
    python main.py \
        --model paligemma \
        --model_name "${MODEL_PALIGEMMA}" \
        --dataset "${dataset}" \
        --algorithm "${QD_ALGO}" \
        --iterations ${ITERATIONS} \
        --exp_name "${EXP_NAME}" \
        ${UNIFIED_FLAGS}
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed PaliGemma on ${dataset}"
        echo "  Results saved to: results/${QD_ALGO}/paligemma_google_paligemma-3b-pt-224/${dataset}/${EXP_NAME}/"
    else
        echo "✗ Failed: PaliGemma on ${dataset}"
    fi
done

# Run Moondream2 experiments
echo ""
echo "========== Moondream2 Experiments =========="
for dataset in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[${CURRENT}/${TOTAL_EXPERIMENTS}] Running Moondream2 on ${dataset}..."
    echo "Command: python main.py --model moondream2 --model_name ${MODEL_MOONDREAM2} --dataset ${dataset} --algorithm ${QD_ALGO} --iterations ${ITERATIONS} --exp_name ${EXP_NAME} ${UNIFIED_FLAGS}"
    
    python main.py \
        --model moondream2 \
        --model_name "${MODEL_MOONDREAM2}" \
        --dataset "${dataset}" \
        --algorithm "${QD_ALGO}" \
        --iterations ${ITERATIONS} \
        --exp_name "${EXP_NAME}" \
        ${UNIFIED_FLAGS}
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed Moondream2 on ${dataset}"
        echo "  Results saved to: results/${QD_ALGO}/moondream2_vikhyatk_moondream2/${dataset}/${EXP_NAME}/"
    else
        echo "✗ Failed: Moondream2 on ${dataset}"
    fi
done

echo ""
echo "========================================"
echo "Completed Algorithm: ${QD_ALGO}"
echo "=========================================="

done  # End algorithm loop

echo ""
echo "========================================"
echo "All experiments complete!"
echo "=========================================="
echo ""
echo "Results are organized in: /root/ICAT/results/"
echo "  - Each algorithm has its own folder"
echo "  - Each model within algorithm folder"
echo "  - Each dataset within model folder"
echo "  - Each experiment within dataset folder"
echo ""
echo "Example structure:"
echo "  results/"
echo "    └── ${QD_ALGO}/"
echo "        ├── blip2_Salesforce_blip2-opt-2.7b/"
echo "        │   ├── ktvic/${EXP_NAME}/"
echo "        │   ├── uit-viic/${EXP_NAME}/"
echo "        │   └── flickr30k/${EXP_NAME}/"
echo "        ├── paligemma_google_paligemma-3b-pt-224/"
echo "        │   ├── ktvic/${EXP_NAME}/"
echo "        │   ├── uit-viic/${EXP_NAME}/"
echo "        │   └── flickr30k/${EXP_NAME}/"
echo "        └── moondream2_vikhyatk_moondream2/"
echo "            ├── ktvic/${EXP_NAME}/"
echo "            ├── uit-viic/${EXP_NAME}/"
echo "            └── flickr30k/${EXP_NAME}/"
echo ""
echo "Checking result directories..."
find results/${QD_ALGO} -type f -name "*.png" -o -name "*.txt" -o -name "*.pkl" 2>/dev/null | head -20
