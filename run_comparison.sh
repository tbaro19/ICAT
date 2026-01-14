#!/bin/bash
# Script to run all QD algorithm comparisons

echo "Running QD Algorithm Comparison Experiments"
echo "============================================"

# Create experiment directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_BASE="comparison_${TIMESTAMP}"

# Common parameters
MODEL="siglip"
MODEL_NAME="ViT-B-16-SigLIP"
DATASET="coco"
ITERATIONS=500
MAX_SAMPLES=50
IMAGE_SIZE=128
EPSILON=0.05

echo -e "\nExperiment parameters:"
echo "  Model: ${MODEL}"
echo "  Dataset: ${DATASET}"
echo "  Iterations: ${ITERATIONS}"
echo "  Max Samples: ${MAX_SAMPLES}"
echo ""

# 1. MAP-Elites
echo "[1/4] Running MAP-Elites..."
python main.py \
    --model ${MODEL} \
    --model_name ${MODEL_NAME} \
    --dataset ${DATASET} \
    --algorithm map_elites \
    --iterations ${ITERATIONS} \
    --max_samples ${MAX_SAMPLES} \
    --image_size ${IMAGE_SIZE} \
    --epsilon ${EPSILON} \
    --exp_name "${EXP_BASE}/map_elites"

# 2. CMA-ME
echo -e "\n[2/4] Running CMA-ME..."
python main.py \
    --model ${MODEL} \
    --model_name ${MODEL_NAME} \
    --dataset ${DATASET} \
    --algorithm cma_me \
    --iterations ${ITERATIONS} \
    --max_samples ${MAX_SAMPLES} \
    --image_size ${IMAGE_SIZE} \
    --epsilon ${EPSILON} \
    --exp_name "${EXP_BASE}/cma_me"

# 3. CMA-MAE
echo -e "\n[3/4] Running CMA-MAE..."
python main.py \
    --model ${MODEL} \
    --model_name ${MODEL_NAME} \
    --dataset ${DATASET} \
    --algorithm cma_mae \
    --iterations ${ITERATIONS} \
    --max_samples ${MAX_SAMPLES} \
    --image_size ${IMAGE_SIZE} \
    --epsilon ${EPSILON} \
    --exp_name "${EXP_BASE}/cma_mae"

# 4. CMA-MEGA
echo -e "\n[4/4] Running CMA-MEGA..."
python main.py \
    --model ${MODEL} \
    --model_name ${MODEL_NAME} \
    --dataset ${DATASET} \
    --algorithm cma_mega \
    --iterations ${ITERATIONS} \
    --max_samples ${MAX_SAMPLES} \
    --image_size ${IMAGE_SIZE} \
    --epsilon ${EPSILON} \
    --exp_name "${EXP_BASE}/cma_mega"

echo -e "\n============================================"
echo "All experiments completed!"
echo "Results saved in: /root/ICAT/results/${EXP_BASE}"
echo ""
