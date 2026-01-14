#!/usr/bin/env python
"""
Quick verification script to test all three VLM models with logit-based fitness
"""
import sys
import torch
import numpy as np
from src.models.blip2_wrapper import BLIP2Wrapper
from src.models.paligemma_wrapper import PaliGemmaWrapper
from src.models.moondream2_wrapper import Moondream2Wrapper
from src.attack.logit_fitness import LogitLossFitness

def test_model(model_name, model_wrapper_class):
    """Test a single model with logit-based fitness"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load model
        print(f"Loading {model_name}...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model_wrapper_class(device=device)
        print(f"✓ {model_name} loaded successfully")
        print(f"  Model class: {model.model.__class__.__name__}")
        
        # Create fitness function
        fitness_fn = LogitLossFitness(device=device, chunk_size=1)
        
        # Detect model type
        detected_type = fitness_fn._detect_model_type(model.model)
        print(f"  Detected type: {detected_type}")
        
        # Create dummy data
        batch_size = 2
        dummy_images = np.random.rand(batch_size, 3, 384, 384).astype(np.float32)
        dummy_captions = ["a photo of a cat"] * batch_size
        
        print(f"\nTesting fitness computation...")
        print(f"  Batch size: {batch_size}")
        print(f"  Image shape: {dummy_images.shape}")
        
        # Compute fitness
        fitness_values = fitness_fn.compute_fitness(
            original_captions=dummy_captions,
            perturbed_images=dummy_images,
            vlm_model=model
        )
        
        print(f"\n✓ Fitness computation successful!")
        print(f"  Fitness shape: {fitness_values.shape}")
        print(f"  Fitness values: {fitness_values}")
        print(f"  Mean: {fitness_values.mean():.4f}")
        print(f"  Std: {fitness_values.std():.4f}")
        
        # Verify fitness values are reasonable
        assert len(fitness_values) == batch_size, f"Expected {batch_size} values, got {len(fitness_values)}"
        assert np.all(fitness_values >= 0), "Fitness values should be non-negative (cross-entropy loss)"
        assert np.all(np.isfinite(fitness_values)), "Fitness values should be finite"
        
        print(f"\n✅ {model_name} PASSED all tests")
        return True
        
    except Exception as e:
        print(f"\n❌ {model_name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("VERIFYING ALL VLM MODELS WITH LOGIT-BASED FITNESS")
    print("="*60)
    
    results = {}
    
    # Test BLIP-2
    results['BLIP-2'] = test_model('BLIP-2', BLIP2Wrapper)
    
    # Test PaliGemma
    results['PaliGemma'] = test_model('PaliGemma', PaliGemmaWrapper)
    
    # Test Moondream2
    results['Moondream2'] = test_model('Moondream2', Moondream2Wrapper)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for model_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{model_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL MODELS VERIFIED SUCCESSFULLY!")
        return 0
    else:
        print("\n⚠️  SOME MODELS FAILED VERIFICATION")
        return 1

if __name__ == '__main__':
    sys.exit(main())
