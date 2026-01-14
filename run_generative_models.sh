#!/bin/bash
# Run experiments with BLIP-2 and LLaVA models

# BLIP-2 experiments
echo "Running BLIP-2 experiments..."
python main.py --model blip2 --model_name Salesforce/blip2-opt-2.7b --dataset ktvic --algorithm cma_me --iterations 10 --exp_name blip2_ktvic
python main.py --model blip2 --model_name Salesforce/blip2-opt-2.7b --dataset uit-viic --algorithm cma_me --iterations 10 --exp_name blip2_uitviic
python main.py --model blip2 --model_name Salesforce/blip2-opt-2.7b --dataset flickr30k --algorithm cma_me --iterations 10 --exp_name blip2_flickr30k

# LLaVA experiments (requires more VRAM)
echo "Running LLaVA experiments..."
python main.py --model llava --model_name llava-hf/llava-v1.6-mistral-7b-hf --dataset ktvic --algorithm cma_me --iterations 10 --exp_name llava_ktvic
python main.py --model llava --model_name llava-hf/llava-v1.6-mistral-7b-hf --dataset uit-viic --algorithm cma_me --iterations 10 --exp_name llava_uitviic
python main.py --model llava --model_name llava-hf/llava-v1.6-mistral-7b-hf --dataset flickr30k --algorithm cma_me --iterations 10 --exp_name llava_flickr30k

echo "All experiments completed!"
