#!/usr/bin/env python
"""
Test script for DeepSeek-VL2-Tiny wrapper
Verifies the model loads and can generate captions
"""
import sys
sys.path.insert(0, '/root/ICAT')

import torch
from PIL import Image
import numpy as np
from src.models.deepseek_vl2_wrapper import DeepSeekVL2Wrapper

def test_deepseek():
    print("="*60)
    print("Testing DeepSeek-VL2-Tiny Wrapper")
    print("="*60)
    
    # Initialize model
    print("\n1. Loading model...")
    try:
        model = DeepSeekVL2Wrapper(
            model_name="deepseek-ai/deepseek-vl2-tiny",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Create a test image
    print("\n2. Creating test image...")
    test_img = np.random.rand(224, 224, 3) * 255
    test_img = test_img.astype(np.uint8)
    pil_img = Image.fromarray(test_img)
    print("✓ Test image created")
    
    # Test caption generation
    print("\n3. Testing caption generation...")
    try:
        caption = model.generate_caption(pil_img, max_length=50)
        print(f"✓ Caption generated: '{caption}'")
    except Exception as e:
        print(f"✗ Failed to generate caption: {e}")
        return False
    
    # Test logit extraction
    print("\n4. Testing logit extraction...")
    try:
        logits = model.extract_logits(pil_img)
        print(f"✓ Logits extracted: shape {logits.shape}, dtype {logits.dtype}")
    except Exception as e:
        print(f"✗ Failed to extract logits: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_deepseek()
    sys.exit(0 if success else 1)
