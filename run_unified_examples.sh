#!/bin/bash
# Example commands for Unified Adaptive Adversarial Framework

echo "=========================================="
echo "Unified Framework Example Commands"
echo "=========================================="

# Activate environment
source /root/miniconda3/bin/activate nlp

# Test 1: Quick verification (10 iterations, 1 sample)
echo ""
echo "Test 1: Quick Verification (Moondream, 10 iterations)"
echo "----------------------------------------------"
python main.py \
  --model moondream2 \
  --dataset ktvic \
  --algorithm cma_me \
  --iterations 10 \
  --max_samples 1 \
  --use_unified \
  --use_logit_loss \
  --exp_name test_unified_quick

# Test 2: BLIP-2 with logit-loss (100 iterations)
echo ""
echo "Test 2: BLIP-2 with Logit-Loss (100 iterations)"
echo "----------------------------------------------"
# Uncomment to run:
# python main.py \
#   --model blip2 \
#   --dataset ktvic \
#   --algorithm cma_me \
#   --iterations 100 \
#   --use_unified \
#   --use_logit_loss \
#   --bc_ranges 0.07 0.20 0.10 0.18 \
#   --grid_dims 12 12 \
#   --exp_name blip2_unified_100iter

# Test 3: PaliGemma full run (1000 iterations)
echo ""
echo "Test 3: PaliGemma Full Run (1000 iterations)"
echo "----------------------------------------------"
# Uncomment to run:
# python main.py \
#   --model paligemma \
#   --model_name google/paligemma-3b-pt-224 \
#   --dataset ktvic \
#   --algorithm cma_me \
#   --iterations 1000 \
#   --batch_size 4 \
#   --sigma0 0.02 \
#   --use_unified \
#   --use_logit_loss \
#   --initial_epsilon 0.05 \
#   --bc_ranges 0.07 0.20 0.10 0.18 \
#   --grid_dims 12 12 \
#   --exp_name paligemma_unified_full

# Test 4: Baseline comparison (without unified framework)
echo ""
echo "Test 4: Baseline Comparison"
echo "----------------------------------------------"
# Baseline (old system):
# python main.py --model blip2 --iterations 1000 --exp_name blip2_baseline

# Unified (new system):
# python main.py --model blip2 --iterations 1000 --use_unified --use_logit_loss --exp_name blip2_unified

echo ""
echo "=========================================="
echo "Done! Uncomment commands above to run full experiments."
echo "=========================================="
