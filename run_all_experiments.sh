#!/bin/bash
# Batch script to run all model+dataset combinations
# Quality-Diversity Black-box Attacks on Vision-Language Models
#
# Usage:
#   ./run_all_experiments.sh [iterations] [exp_name]
#
# Examples:
#   ./run_all_experiments.sh                    # 1000 iterations, jailbreak_run
#   ./run_all_experiments.sh 50                 # 50 iterations, jailbreak_run
#   ./run_all_experiments.sh 500 baseline       # 500 iterations, baseline
#   ./run_all_experiments.sh 1000 safety_test   # 1000 iterations, safety_test
#
# Note: Experiments run SEQUENTIALLY (not parallel) - one at a time
#

# Parse command line arguments
ITERATIONS=${1:-1000}
EXP_NAME=${2:-"jailbreak_run"}

# All QD algorithms to run
ALGORITHMS=("cma_me" "cma_mae" "cma_mega")

echo "========================================"
echo "🎯 Safety Red-Teaming Jailbreak Framework"
echo "========================================"
echo "Configuration:"
echo "  - Iterations: ${ITERATIONS}"
echo "  - Experiment name: ${EXP_NAME}"
echo "  - QD Algorithms: cma_me, cma_mae, cma_mega"
echo "  - Mode: Unified Adaptive Framework (Jailbreak)"
echo "  - Fitness: Harmful Token Lexicon (Always Enabled)"
echo "=========================================="

# Always use unified framework with jailbreak fitness
UNIFIED_FLAGS="--use_unified"

# Model configurations (optimized for Tesla T4)
MODEL_DEEPSEEK="deepseek-ai/deepseek-vl2-tiny"
MODEL_INTERNVL="OpenGVLab/InternVL2-2B"
MODEL_QWEN2VL="Qwen/Qwen2-VL-2B-Instruct"

# Datasets
DATASETS=("uit-viic")

# Track progress
TOTAL_EXPERIMENTS=$((2 * ${#DATASETS[@]} * ${#ALGORITHMS[@]}))
CURRENT=0

echo "Total experiments to run: ${TOTAL_EXPERIMENTS}"
echo "Algorithms: ${ALGORITHMS[@]}"
echo "Models: DeepSeek-VL2-Tiny, InternVL2-2B (Qwen2-VL ready)"
echo "Datasets: ${DATASETS[@]}"
echo ""

# Loop through all algorithms
for QD_ALGO in "${ALGORITHMS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Starting Algorithm: ${QD_ALGO}"
    echo "=========================================="
    echo ""

# Run DeepSeek-VL2 experiments
echo "========== DeepSeek-VL2 Experiments =========="
for dataset in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[${CURRENT}/${TOTAL_EXPERIMENTS}] Running DeepSeek-VL2 on ${dataset}..."
    echo "Command: python main.py --model deepseek --model_name ${MODEL_DEEPSEEK} --dataset ${dataset} --algorithm ${QD_ALGO} --iterations ${ITERATIONS} --exp_name ${EXP_NAME} ${UNIFIED_FLAGS}"
    
    python main.py \
        --model deepseek \
        --model_name "${MODEL_DEEPSEEK}" \
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

# Run InternVL2 experiments
echo ""
echo "========== InternVL2-2B Experiments =========="
for dataset in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[${CURRENT}/${TOTAL_EXPERIMENTS}] Running InternVL2 on ${dataset}..."
    echo "Command: python main.py --model internvl --model_name ${MODEL_INTERNVL} --dataset ${dataset} --algorithm ${QD_ALGO} --iterations ${ITERATIONS} --exp_name ${EXP_NAME} ${UNIFIED_FLAGS}"
    
    python main.py \
        --model internvl \
        --model_name "${MODEL_INTERNVL}" \
        --dataset "${dataset}" \
        --algorithm "${QD_ALGO}" \
        --iterations ${ITERATIONS} \
        --exp_name "${EXP_NAME}" \
        ${UNIFIED_FLAGS}
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed InternVL2 on ${dataset}"
        echo "  Results saved to: results/${QD_ALGO}/internvl_OpenGVLab_InternVL2-2B/${dataset}/${EXP_NAME}/"
    else
        echo "✗ Failed: InternVL2 on ${dataset}"
    fi
done

# Run Qwen2-VL experiments (OPTIONAL - uncomment to enable)
# echo ""
# echo "========== Qwen2-VL Experiments =========="
# for dataset in "${DATASETS[@]}"; do
#     CURRENT=$((CURRENT + 1))
#     echo ""
#     echo "[${CURRENT}/${TOTAL_EXPERIMENTS}] Running Qwen2-VL on ${dataset}..."
#     echo "Command: python main.py --model qwen2vl --model_name ${MODEL_QWEN2VL} --dataset ${dataset} --algorithm ${QD_ALGO} --iterations ${ITERATIONS} --exp_name ${EXP_NAME} ${UNIFIED_FLAGS}"
#     
#     python main.py \
#         --model qwen2vl \
#         --model_name "${MODEL_QWEN2VL}" \
#         --dataset "${dataset}" \
#         --algorithm "${QD_ALGO}" \
#         --iterations ${ITERATIONS} \
#         --exp_name "${EXP_NAME}" \
#         ${UNIFIED_FLAGS}
#     
#     if [ $? -eq 0 ]; then
#         echo "✓ Successfully completed Qwen2-VL on ${dataset}"
#         echo "  Results saved to: results/${QD_ALGO}/qwen2vl_Qwen_Qwen2-VL-2B-Instruct/${dataset}/${EXP_NAME}/"
#     else
#         echo "✗ Failed: Qwen2-VL on ${dataset}"
#     fi
# done

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
echo "        ├── internvl_OpenGVLab_InternVL2-2B/"
echo "        │   ├── ktvic/${EXP_NAME}/"
echo "        │   ├── uit-viic/${EXP_NAME}/"
echo "        │   └── flickr30k/${EXP_NAME}/"
echo "        └── qwen2vl_Qwen_Qwen2-VL-2B-Instruct/"
echo "            ├── ktvic/${EXP_NAME}/"
echo "            ├── uit-viic/${EXP_NAME}/"
echo "            └── flickr30k/${EXP_NAME}/"
echo ""
echo "Checking result directories..."
find results/${QD_ALGO} -type f -name "*.png" -o -name "*.txt" -o -name "*.pkl" 2>/dev/null | head -20
